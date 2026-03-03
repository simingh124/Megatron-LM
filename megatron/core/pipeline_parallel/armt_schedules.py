"""ARMT TBPTT schedule for no-pipeline training."""

import contextlib
from typing import Dict, Iterator, List, Optional

import torch

from megatron.core import parallel_state
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.utils import get_model_config, get_model_type, unwrap_model

from .schedules import backward_step


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
            raise NotImplementedError("ARMT v1 does not support packed sequences")

        chunks.append(chunk)
    return chunks


def _get_num_tokens(chunk: Dict) -> float:
    if "loss_mask" in chunk and isinstance(chunk["loss_mask"], torch.Tensor):
        return float(chunk["loss_mask"].float().sum().item())
    if "tokens" in chunk and isinstance(chunk["tokens"], torch.Tensor):
        return float(chunk["tokens"].numel())
    return 0.0


def armt_forward_backward_no_pipelining(
    *,
    forward_step_func,
    data_iterator: Iterator,
    model,
    num_microbatches: int,
    chunk_size: Optional[int] = None,
    seq_length: int,
    decoder_seq_length: Optional[int] = None,  # compatibility with training loop
    encoder_seq_length: Optional[int] = None,  # unused (decoder-only)
    micro_batch_size: int,  # unused
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: Optional[bool] = None,
    pg_collection: Optional[ProcessGroupCollection] = None,
    force_all_reduce: Optional[bool] = False,
    **_unused_kwargs,
):
    if collect_non_loss_data:
        raise NotImplementedError("ARMT TBPTT schedule does not support collect_non_loss_data")

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

    losses_reduced = []
    total_num_tokens = torch.zeros([], dtype=torch.int)

    from megatron.training import global_vars as training_global_vars

    if chunk_size is None:
        if training_global_vars._GLOBAL_ARGS is None:
            raise ValueError("ARMT TBPTT requires armt_chunk_size to be set in args")
        chunk_size = training_global_vars.get_args().armt_chunk_size

    # Some training loops pass `decoder_seq_length` explicitly (e.g., dual-stack models).
    # ARMT is decoder-only; use decoder_seq_length if provided for compatibility.
    if decoder_seq_length is not None:
        seq_length = decoder_seq_length

    for microbatch_id in range(num_microbatches):
        # Note: in Megatron pretrain flows, only TP rank 0 may have a real
        # data iterator; other TP ranks can receive the batch via broadcast.
        # Use the standard helper to make this schedule compatible with TP>1.
        from megatron.training.utils import get_batch_on_this_tp_rank

        raw_batch = get_batch_on_this_tp_rank(data_iterator)
        # ARMT v1 does not support packed sequences; keep explicit key for forward_step.
        raw_batch["packed_seq_params"] = None
        chunks = chunk_data(raw_batch, chunk_size, seq_length)
        num_chunks = len(chunks)

        args = training_global_vars.get_args() if training_global_vars._GLOBAL_ARGS is not None else None
        if args is not None and getattr(args, "no_loss_from_first_chunk", False) and num_chunks > 0:
            # Implement --no-loss-from-first-chunk by masking the first chunk's loss.
            # Forward still runs (memory updates happen in forward).
            if "loss_mask" not in chunks[0] or not isinstance(chunks[0]["loss_mask"], torch.Tensor):
                raise ValueError(
                    "--no-loss-from-first-chunk requires batch['loss_mask'] to exist and be a tensor."
                )
            chunks[0]["loss_mask"] = torch.zeros_like(chunks[0]["loss_mask"])

        unwrapped_model = unwrap_model(model)
        if hasattr(unwrapped_model, "reset_all_memory"):
            unwrapped_model.reset_all_memory()

        total_tokens_this_micro = sum(_get_num_tokens(c) for c in chunks)
        total_tokens_this_micro = max(total_tokens_this_micro, 1.0)

        for chunk_idx, chunk in enumerate(chunks):
            chunk_tokens = _get_num_tokens(chunk)
            loss_weight = (chunk_tokens / total_tokens_this_micro) * (1.0 / num_microbatches)

            chunk_iter = iter([chunk])

            def run_chunk():
                output_tensor, loss_func = forward_step_func(chunk_iter, model)
                if loss_func is None:
                    if chunk_idx == num_chunks - 1:
                        losses_reduced.append(output_tensor.detach())
                    return

                outputs = loss_func(output_tensor)
                if len(outputs) == 3:
                    loss, num_tokens, loss_reduced = outputs
                    if not config.calculate_per_token_loss:
                        loss /= torch.clamp(num_tokens, min=1)
                else:
                    loss, loss_reduced = outputs
                    loss *= pg_collection.cp.size()

                scaled_loss = loss * loss_weight

                # Optimization: if this chunk carries no loss (e.g. masked), skip backward_step.
                if not forward_only and chunk_tokens > 0.0 and loss_weight > 0.0:
                    backward_step(None, scaled_loss, None, model_type, config)

                if chunk_idx == num_chunks - 1:
                    losses_reduced.append(loss_reduced)

            is_last_microbatch = microbatch_id == num_microbatches - 1
            is_last_chunk = chunk_idx == num_chunks - 1

            if not is_last_microbatch or not is_last_chunk:
                with no_sync_func():
                    run_chunk()
            else:
                run_chunk()

            total_num_tokens += int(chunk_tokens)

    if config.finalize_model_grads_func is not None and not forward_only:
        config.finalize_model_grads_func(
            [model],
            total_num_tokens if config.calculate_per_token_loss else None,
            pg_collection=pg_collection,
            force_all_reduce=force_all_reduce,
        )

    return losses_reduced
