"""Shared recurrent TBPTT schedule for no-pipeline training."""

import contextlib
from typing import Dict, Iterator, List, Optional, Tuple

import torch

from megatron.core import parallel_state
from megatron.core.models.armt.monitoring import (
    build_ratio_metric,
    clear_armt_tensorboard_metrics,
    merge_metric_primitives,
    publish_armt_tensorboard_metrics,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.utils import get_model_config, get_model_type, unwrap_model

from .schedules import backward_step

_PRIMARY_LOSS_KEY = "lm loss"


def _get_recurrent_chunk_size(args) -> Optional[int]:
    chunk_size = getattr(args, "recurrent_chunk_size", None)
    if chunk_size is not None:
        return chunk_size
    return getattr(args, "armt_chunk_size", None)


def _is_sequence_tensor(tensor: torch.Tensor, seq_length: int) -> Optional[int]:
    if tensor.dim() >= 2 and tensor.shape[1] == seq_length:
        return 1
    if tensor.dim() >= 2 and tensor.shape[0] == seq_length:
        return 0
    return None


def _slice_sequence_tensor(
    tensor: torch.Tensor, seq_dim: int, start_idx: int, end_idx: int
) -> torch.Tensor:
    if seq_dim == 0:
        return tensor[start_idx:end_idx].contiguous()
    if seq_dim == 1:
        return tensor[:, start_idx:end_idx].contiguous()
    return tensor


def chunk_data(data: Dict, chunk_size: int, seq_length: int) -> List[Dict]:
    chunks = []
    for start_idx in range(0, seq_length, chunk_size):
        end_idx = min(start_idx + chunk_size, seq_length)
        chunk = {}
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                seq_dim = _is_sequence_tensor(value, seq_length)
                if seq_dim is not None:
                    chunk[key] = _slice_sequence_tensor(value, seq_dim, start_idx, end_idx)
                elif value.dim() >= 4 and value.shape[-1] == seq_length and value.shape[-2] == seq_length:
                    chunk[key] = value[..., start_idx:end_idx, start_idx:end_idx].contiguous()
                else:
                    chunk[key] = value
            else:
                chunk[key] = value

        if "packed_seq_params" in chunk and chunk["packed_seq_params"] is not None:
            raise NotImplementedError("Recurrent TBPTT v1 does not support packed sequences")

        chunks.append(chunk)
    return chunks


def _get_num_tokens(chunk: Dict) -> float:
    if "loss_mask" in chunk and isinstance(chunk["loss_mask"], torch.Tensor):
        return float(chunk["loss_mask"].float().sum().item())
    if "tokens" in chunk and isinstance(chunk["tokens"], torch.Tensor):
        return float(chunk["tokens"].numel())
    return 0.0


def _accumulate_chunk_reports(
    accumulated: Optional[Dict[str, torch.Tensor]],
    scalar_accumulators: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    chunk_report: Dict[str, torch.Tensor],
    *,
    chunk_tokens: float,
) -> Dict[str, torch.Tensor]:
    if accumulated is None:
        accumulated = {}

    for key, value in chunk_report.items():
        if not isinstance(value, torch.Tensor):
            raise ValueError(
                "Recurrent TBPTT schedule expects loss_reduced dict values to be torch.Tensors "
                f"(got key={key}, type={type(value)})."
            )
        if value.numel() == 2:
            if key not in accumulated:
                accumulated[key] = value.clone()
            else:
                accumulated[key] = accumulated[key] + value
        elif value.numel() == 1:
            if chunk_tokens <= 0.0:
                continue
            token_tensor = value.new_tensor(chunk_tokens)
            if key not in scalar_accumulators:
                scalar_accumulators[key] = (value.clone() * token_tensor, token_tensor)
            else:
                weighted_sum, token_sum = scalar_accumulators[key]
                scalar_accumulators[key] = (
                    weighted_sum + value * token_tensor,
                    token_sum + token_tensor,
                )
        else:
            raise ValueError(
                "Recurrent TBPTT schedule expects loss_reduced dict values to have "
                f"shape (1,) or (2,) (got key={key}, shape={tuple(value.shape)})."
            )

    return accumulated


def _accumulate_chunk_loss_metrics(
    per_chunk_loss_sums: Dict[int, torch.Tensor],
    per_chunk_token_sums: Dict[int, torch.Tensor],
    *,
    chunk_idx: int,
    chunk_report: Dict[str, torch.Tensor],
) -> None:
    loss_report = chunk_report.get(_PRIMARY_LOSS_KEY)
    if not isinstance(loss_report, torch.Tensor) or loss_report.numel() != 2:
        return

    loss_report = loss_report.view(-1).detach().to(dtype=torch.float32)
    current_loss_sum = per_chunk_loss_sums.get(chunk_idx)
    current_token_sum = per_chunk_token_sums.get(chunk_idx)
    if current_loss_sum is None:
        per_chunk_loss_sums[chunk_idx] = loss_report[0]
        per_chunk_token_sums[chunk_idx] = loss_report[1]
        return

    per_chunk_loss_sums[chunk_idx] = current_loss_sum + loss_report[0]
    per_chunk_token_sums[chunk_idx] = current_token_sum + loss_report[1]


def _set_recurrent_chunk_model_state(model, *, args, is_first_chunk: bool) -> None:
    if hasattr(model, "set_current_chunk_is_first"):
        model.set_current_chunk_is_first(is_first_chunk)

    if hasattr(model, "set_skip_read_memory_for_current_chunk"):
        skip_read_memory = bool(
            args is not None
            and is_first_chunk
            and getattr(args, "no_read_memory_from_first_chunk", False)
        )
        model.set_skip_read_memory_for_current_chunk(skip_read_memory)


def _backward_full_microbatch_loss(loss: torch.Tensor, config) -> None:
    """Backward the accumulated recurrent loss once per microbatch in no-TBPTT mode."""
    if config.timers is not None:
        config.timers("backward-compute", log_level=2).start()

    if config.grad_scale_func is not None:
        loss = config.grad_scale_func(loss)

    if loss.requires_grad:
        torch.autograd.backward(loss)

    if config.timers is not None:
        config.timers("backward-compute").stop()


def recurrent_forward_backward_no_pipelining(
    *,
    forward_step_func,
    data_iterator: Iterator,
    model,
    num_microbatches: int,
    chunk_size: Optional[int] = None,
    seq_length: int,
    decoder_seq_length: Optional[int] = None,
    encoder_seq_length: Optional[int] = None,
    micro_batch_size: int,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: Optional[bool] = None,
    pg_collection: Optional[ProcessGroupCollection] = None,
    force_all_reduce: Optional[bool] = False,
    **_unused_kwargs,
):
    del encoder_seq_length, micro_batch_size, first_val_step
    if collect_non_loss_data:
        raise NotImplementedError("Recurrent TBPTT schedule does not support collect_non_loss_data")

    if pg_collection is None:
        tp_group = parallel_state.get_tensor_model_parallel_group()
        cp_group = parallel_state.get_context_parallel_group()
        embd_group = parallel_state.get_embedding_group(check_initialized=False)
        pp_group = parallel_state.get_pipeline_model_parallel_group()
        pos_emb_group = parallel_state.get_position_embedding_group(check_initialized=False)
        pg_collection = ProcessGroupCollection()
        pg_collection.tp = tp_group
        pg_collection.cp = cp_group
        pg_collection.embd = embd_group
        pg_collection.pos_embd = pos_emb_group
        pg_collection.pp = pp_group
        pg_collection.dp_cp = parallel_state.get_data_parallel_group(
            with_context_parallel=True, partial_data_parallel=False
        )

    if isinstance(model, list):
        assert len(model) == 1
        model = model[0]
    if isinstance(data_iterator, list):
        assert len(data_iterator) == 1
        data_iterator = data_iterator[0]

    config = get_model_config(model)
    model_type = get_model_type(model)

    no_sync_func = config.no_sync_func
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext

    clear_armt_tensorboard_metrics()
    losses_reduced = []
    total_num_tokens = torch.zeros([], dtype=torch.int)
    per_chunk_loss_sums: Dict[int, torch.Tensor] = {}
    per_chunk_token_sums: Dict[int, torch.Tensor] = {}
    unwrapped_model = unwrap_model(model)
    if hasattr(unwrapped_model, "reset_all_monitoring_stats"):
        unwrapped_model.reset_all_monitoring_stats()

    from megatron.training import global_vars as training_global_vars

    if chunk_size is None:
        if training_global_vars._GLOBAL_ARGS is None:
            raise ValueError("Recurrent TBPTT requires recurrent_chunk_size to be set in args")
        chunk_size = _get_recurrent_chunk_size(training_global_vars.get_args())
        if chunk_size is None:
            raise ValueError("Recurrent TBPTT requires recurrent_chunk_size to be set in args")

    if decoder_seq_length is not None:
        seq_length = decoder_seq_length

    for microbatch_id in range(num_microbatches):
        from megatron.training.utils import get_batch_on_this_tp_rank

        raw_batch = get_batch_on_this_tp_rank(data_iterator)
        raw_batch["packed_seq_params"] = None
        chunks = chunk_data(raw_batch, chunk_size, seq_length)
        num_chunks = len(chunks)

        args = training_global_vars.get_args() if training_global_vars._GLOBAL_ARGS is not None else None
        recurrent_tbptt_mode = True
        if args is not None:
            recurrent_tbptt_mode = getattr(
                args,
                "recurrent_tbptt_mode",
                getattr(args, "armt_tbptt_mode", True),
            )
        if args is not None and getattr(args, "no_loss_from_first_chunk", False) and num_chunks > 0:
            if "loss_mask" not in chunks[0] or not isinstance(chunks[0]["loss_mask"], torch.Tensor):
                raise ValueError(
                    "--no-loss-from-first-chunk requires batch['loss_mask'] to exist and be a tensor."
                )
            chunks[0]["loss_mask"] = torch.zeros_like(chunks[0]["loss_mask"])

        if hasattr(unwrapped_model, "reset_all_memory"):
            unwrapped_model.reset_all_memory()

        total_tokens_this_micro = sum(_get_num_tokens(c) for c in chunks)
        total_tokens_this_micro = max(total_tokens_this_micro, 1.0)

        microbatch_loss_reduced: Optional[Dict[str, torch.Tensor]] = None
        microbatch_scalar_accumulators: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        microbatch_total_loss: Optional[torch.Tensor] = None
        last_chunk_output_tensor: Optional[torch.Tensor] = None

        def run_chunk(chunk_idx: int, chunk: Dict) -> float:
            nonlocal last_chunk_output_tensor
            nonlocal microbatch_loss_reduced
            nonlocal microbatch_total_loss

            chunk_tokens = _get_num_tokens(chunk)
            if config.calculate_per_token_loss:
                loss_weight = 1.0
            else:
                loss_weight = (chunk_tokens / total_tokens_this_micro) * (1.0 / num_microbatches)

            chunk_iter = iter([chunk])

            output_tensor, loss_func = forward_step_func(chunk_iter, model)
            if loss_func is None:
                if chunk_idx == num_chunks - 1:
                    last_chunk_output_tensor = output_tensor.detach()
                return chunk_tokens

            outputs = loss_func(output_tensor)
            if len(outputs) == 3:
                loss, num_tokens, loss_reduced = outputs
                if not config.calculate_per_token_loss:
                    loss /= torch.clamp(num_tokens, min=1)
            else:
                loss, loss_reduced = outputs
                loss *= pg_collection.cp.size()

            scaled_loss = loss * loss_weight

            if not forward_only and chunk_tokens > 0.0:
                if recurrent_tbptt_mode:
                    backward_step(None, scaled_loss, None, model_type, config)
                elif microbatch_total_loss is None:
                    microbatch_total_loss = scaled_loss
                else:
                    microbatch_total_loss = microbatch_total_loss + scaled_loss

            if not isinstance(loss_reduced, dict):
                raise ValueError(
                    "Recurrent TBPTT schedule expects loss_func to return a dict as "
                    f"loss_reduced (got {type(loss_reduced)})."
                )
            _accumulate_chunk_loss_metrics(
                per_chunk_loss_sums,
                per_chunk_token_sums,
                chunk_idx=chunk_idx,
                chunk_report=loss_reduced,
            )
            microbatch_loss_reduced = _accumulate_chunk_reports(
                microbatch_loss_reduced,
                microbatch_scalar_accumulators,
                loss_reduced,
                chunk_tokens=chunk_tokens,
            )

            return chunk_tokens

        def run_chunk_with_model_state(chunk_idx: int, chunk: Dict) -> float:
            _set_recurrent_chunk_model_state(
                unwrapped_model,
                args=args,
                is_first_chunk=chunk_idx == 0,
            )
            try:
                return run_chunk(chunk_idx, chunk)
            finally:
                _set_recurrent_chunk_model_state(
                    unwrapped_model,
                    args=None,
                    is_first_chunk=False,
                )

        is_last_microbatch = microbatch_id == num_microbatches - 1

        if recurrent_tbptt_mode:
            for chunk_idx, chunk in enumerate(chunks):
                is_last_chunk = chunk_idx == num_chunks - 1
                if not is_last_microbatch or not is_last_chunk:
                    with no_sync_func():
                        chunk_tokens = run_chunk_with_model_state(chunk_idx, chunk)
                else:
                    chunk_tokens = run_chunk_with_model_state(chunk_idx, chunk)

                total_num_tokens += int(chunk_tokens)
        else:
            microbatch_context = (
                no_sync_func() if not forward_only and not is_last_microbatch else contextlib.nullcontext()
            )
            with microbatch_context:
                for chunk_idx, chunk in enumerate(chunks):
                    chunk_tokens = run_chunk_with_model_state(chunk_idx, chunk)
                    total_num_tokens += int(chunk_tokens)

                if not forward_only and microbatch_total_loss is not None:
                    _backward_full_microbatch_loss(microbatch_total_loss, config)
                    microbatch_total_loss = None

        if microbatch_loss_reduced is not None:
            for key, (weighted_sum, token_sum) in microbatch_scalar_accumulators.items():
                microbatch_loss_reduced[key] = weighted_sum / torch.clamp(token_sum, min=1)
            losses_reduced.append(microbatch_loss_reduced)
        elif last_chunk_output_tensor is not None:
            losses_reduced.append(last_chunk_output_tensor)

    if config.finalize_model_grads_func is not None and not forward_only:
        config.finalize_model_grads_func(
            [model],
            total_num_tokens if config.calculate_per_token_loss else None,
            pg_collection=pg_collection,
            force_all_reduce=force_all_reduce,
        )

    monitoring_primitives = {}
    if hasattr(unwrapped_model, "consume_all_monitoring_primitives"):
        model_primitives = unwrapped_model.consume_all_monitoring_primitives()
        if isinstance(model_primitives, dict):
            merge_metric_primitives(
                monitoring_primitives,
                model_primitives,
            )

    if not forward_only:
        for chunk_idx, loss_sum in per_chunk_loss_sums.items():
            metric_name = f"train/chunk_{chunk_idx:02d}_loss"
            monitoring_primitives[metric_name] = build_ratio_metric(
                loss_sum,
                per_chunk_token_sums[chunk_idx],
            )
        publish_armt_tensorboard_metrics(monitoring_primitives)

    return losses_reduced

__all__ = [
    "chunk_data",
    "recurrent_forward_backward_no_pipelining",
]
