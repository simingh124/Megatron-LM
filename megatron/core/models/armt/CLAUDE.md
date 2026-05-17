# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

This is a **submodule-scoped** CLAUDE.md for `megatron/core/models/armt/`. The repository-root `CLAUDE.md` already covers project-wide commands, ARMT CLI arguments, parallelism constraints, and test entry points — read it first. The notes here only describe internals of this directory that are not derivable from a quick read of the code.

## Module Purpose

Provides the ARMT (Associative Recurrent Memory Transformer) building blocks: an `ARMTModel` that wraps `GPTModel` with per-layer memory tokens, an `ARMTLayer` that performs retrieval → attention/MLP → memory write each step, and a pluggable family of recurrent-memory backends behind a single backend selector.

## Pluggable Recurrent-Memory Backend

`ARMTLayer` does not hard-wire `AssociativeLayer` — it instantiates one of three backends via `recurrent_memory.build_recurrent_memory_backend(backend, **kwargs)`. The backend is chosen by the `recurrent_memory_backend` parameter passed through `get_armt_layer_spec(...)`.

| `recurrent_memory_backend` | Class | File | Notes |
| --- | --- | --- | --- |
| `"associative"` (default) | `AssociativeLayer` | `associative_layer.py` | DPFP-based key/value with optional denom (`z`), gating, correction; supports TBPTT detach |
| `"gated_deltanet"` | `GatedDeltaNetMemory` | `gated_deltanet_memory.py` | FLA-kernel / causal-conv1d delta-net memory; takes `recurrent_gdn_*` kwargs |
| `"cross_attn_slots"` | `CrossAttentionSlotMemory` | `cross_attention_slot_memory.py` | Learnable slot bank read via cross-attention (`recurrent_slot_*` kwargs); `read_attn_backend` ∈ {`"flash"`, …} |

All backends implement the same protocol used by `ARMTLayer`:

- `associate(hidden_states, *, input_is_sbh) -> retrieved` — produces a residual added before attention. May early-return zeros (e.g. first chunk).
- `update_mem(write_part, *, input_is_sbh, input_already_pre_normed)` — consumes either memory tokens or a context slice (see below) and mutates internal state in place.
- `reset_memory()` — clear per-sequence state. Called by `ARMTModel.reset_all_memory()` (and by `ARMTLayer.reset_memory()`), which fans out to every `ARMTLayer`.
- `reset_monitoring_stats()` / `consume_monitoring_primitives()` — see Monitoring below.

When adding a new backend: register it in `SUPPORTED_RECURRENT_MEMORY_BACKENDS` and `build_recurrent_memory_backend` in `recurrent_memory.py`, and add a corresponding branch in `ARMTLayer.__init__` that forwards the right kwargs. The `common_kwargs` block in `armt_layer.py` is the canonical contract every backend constructor must accept (`config`, `d_model`, `num_mem_tokens`, `tbptt_mode`, `normalization`, `norm_epsilon`, `use_input_pre_norm`, `log_read_position_metrics_to_tensorboard`).

## Memory Write Source (`armt_memory_write_source`)

`ARMTLayer` decouples *what gets written* into the recurrent state from the memory-token slot. Configured by `armt_memory_write_source` ∈ `{"mem_tokens", "pre_attn_context", "post_attn_context", "post_mlp_context"}`:

- `mem_tokens` (default) — slice the last `num_mem_tokens` positions of the post-MLP hidden state.
- `post_mlp_context` / `post_attn_context` — slice the context (non-memory) portion of the corresponding hidden state.
- `pre_attn_context` — write from the layer input (optionally post-`input_layernorm`).

The `pre_attn_context + recurrent_memory_input_pre_norm=True` path captures the LayerNorm output via a forward hook installed in `ARMTLayer.__init__` (`_capture_input_layernorm_output`). If the capture is empty (e.g. the LayerNorm was bypassed in cudagraph mode), it logs a one-shot warning and falls back to recomputing the norm. When the captured value is used, `input_already_pre_normed=True` is signaled to the backend so it does not double-norm.

## Data Flow (Quick Reference)

`ARMTModel.forward`:
1. `GPTModel._preprocess` → `decoder_input` (SBH layout).
2. Append `memory_embeddings` along the sequence dim (S → S+M) and extend the attention mask + RoPE/Yarn embeddings to `S+M`.
3. Run `self.decoder` (every layer is an `ARMTLayer`).
4. Strip the last `M` positions back off (S+M → S) before `_postprocess` (LM head, loss).

`ARMTLayer.forward`:
1. `associate(hidden_states)` → residual add (skipped when `_skip_read_memory_for_current_chunk` is set, used for the first TBPTT chunk).
2. Standard `_forward_attention` + `_forward_mlp`.
3. Resolve the write source per `armt_memory_write_source`, then call `update_mem(...)`.
4. Accumulate token-level monitoring stats (norms, mem-token cosine spread) when collection is enabled.

## Sequence Layout / Parallelism

- **SBH only.** `ARMTLayer.forward` hard-codes `input_is_sbh = True`. Splitting on the sequence dim assumes `[S, B, H]`. Do not pass BSH.
- **Sequence parallel.** When `config.sequence_parallel` is true, `_prepare_hidden_states_for_memory_ops` does a `gather_from_sequence_parallel_region` so memory ops see the full sequence. Anything you add that touches `hidden_states[-num_mem_tokens:]` must go through this helper.
- **TP gather for monitoring.** `_gather_hidden_for_monitoring` checks the trailing dim against `config.hidden_size` and `gather_from_tensor_model_parallel_region`s if the tensor is still TP-sharded. Use it whenever metrics need full hidden vectors.
- **Packed sequences are not supported.** `PackedSeqParams` must be `None`.

## Monitoring / TensorBoard Metrics

`monitoring.py` defines a small reduce-friendly metric primitive system used to collect ARMT-specific TensorBoard metrics across ranks:

- `MetricPrimitive(kind, numerator, denominator)` with kinds `mean`, `ratio`, `ratio_of_means`, `rms`.
- `build_mean_metric` / `build_ratio_metric` / `build_ratio_of_means_metric` / `build_rms_metric` are the only constructors layers should use — they normalize the inputs to scalar fp32 tensors.
- `accumulate_armt_tensorboard_metrics` / `publish_armt_tensorboard_metrics` write into a module-level tracker.
- `consume_armt_tensorboard_metrics(reduce_group, finalize_on_this_rank)` flushes the tracker, all-reduces the flat buffers per device, and returns the finalized scalar tensors on the chosen rank.

`ARMTLayer.consume_monitoring_primitives()` produces metrics under the `armt/token/*` namespace (mem-token norms, context-token norms, mem-token cosine spread) and merges in whatever the active backend exposes via its own `consume_monitoring_primitives()`. Backends are expected to follow the same merge protocol.

Collection is per-iteration: `ARMTLayer.set_collect_monitoring_for_current_iteration(enabled)` toggles both the layer and the backend; the TBPTT schedule typically enables it only on the last chunk to avoid double-counting.

## Init and Norm Helpers

- `init_utils.init_parameter` / `init_linear_weight_and_bias` — the standard way for ARMT submodules to honor `config.perform_initialization` (in distributed checkpoint flows, weights may already be loaded and must not be re-initialized). New tensors added inside a backend should go through these helpers rather than calling `init_method` directly.
- `norm_utils.build_recurrent_norm(hidden_size, normalization=..., eps=..., dtype=...)` — single entry point for `LayerNorm` / `RMSNorm`. Any new backend should use this so the `--normalization` CLI flag continues to drive every recurrent norm uniformly.

## Checkpoint Naming Convention

ARMT state-dict keys (must stay stable for `tools/convert_baseline_to_armt.py` and integration tests):

- Top-level: `memory_embeddings`
- Per layer: `decoder.layers.{i}.recurrent_memory_layer.*`

Runtime-only buffers (e.g. `W_mem`, `z` in `AssociativeLayer`) must be excluded from the checkpoint. If you add new persistent parameters to a backend, mirror the existing exclusion logic and update the conversion tool plus `examples/armt/tests/test_armt_checkpoint.py`.

## TBPTT Hooks Exposed by `ARMTLayer`

The TBPTT schedule (`megatron/core/pipeline_parallel/armt_schedules.py`) drives ARMT layers chunk-by-chunk via these setters:

- `set_current_chunk_is_first(bool)` — informs the layer that this chunk is the first of a sequence (some backends use it to gate first-chunk behavior).
- `set_skip_read_memory_for_current_chunk(bool)` — when true, `associate` is skipped (no retrieval residual added). Used on the first chunk.
- `set_collect_monitoring_for_current_iteration(bool)` — restricts metric accumulation to a single chunk per iteration.
- `reset_memory()` — clears layer-side flags and delegates to the backend.

When extending TBPTT semantics, prefer adding more setters on `ARMTLayer` rather than reaching into the backend from the schedule directly.

## AGENTS.md

A shorter Chinese-language version of this doc exists at `AGENTS.md` in the same directory; keep them broadly consistent when changing the architecture.
