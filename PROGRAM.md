# PROGRAM

## Purpose
- This file stores repo-specific operational knowledge that is easy to forget and expensive to rediscover: branch/worktree topology, hidden coupling, test-double invariants, and doc/code drift traps.
- Stable always-on rules belong in `AGENTS.md`; do not duplicate them here unless the repo has a concrete exception or counterexample.

## Branch and Worktree Topology
- `armt` is the integration branch in the root worktree `/mnt/step3-abla/siming/code_repo/Megatron-LM`.
- `concat` lives in sibling worktree `/mnt/step3-abla/siming/code_repo/Megatron-LM-concat` and currently starts from `armt` commit `e1306a42f`. If it needs to be recreated from a newer `armt`, remove the sibling worktree first and then create the branch/worktree explicitly from the refreshed `armt` HEAD.
- `lr` lives in sibling worktree `/mnt/step3-abla/siming/code_repo/Megatron-LM-lr` and currently starts from `armt` commit `e5549212d`. If it needs to be recreated from a newer `armt`, remove the sibling worktree first and then create the branch/worktree explicitly from the refreshed `armt` HEAD.
- `metric` lives in sibling worktree `/mnt/step3-abla/siming/code_repo/Megatron-LM-metric` and currently starts from `armt` commit `e5549212d`. If it needs to be recreated from a newer `armt`, remove the sibling worktree first and then create the branch/worktree explicitly from the refreshed `armt` HEAD.
- `omit` lives in sibling worktree `/mnt/step3-abla/siming/code_repo/Megatron-LM-omit` and currently starts from `armt` commit `e5549212d`. If it needs to be recreated from a newer `armt`, remove the sibling worktree first and then create the branch/worktree explicitly from the refreshed `armt` HEAD.
- `wofst` lives in sibling worktree `/mnt/step3-abla/siming/code_repo/Megatron-LM-wofst` and currently starts from `armt` commit `b2b06ade7`. If it needs to be recreated from a newer `armt`, remove the sibling worktree first and then create the branch/worktree explicitly from the refreshed `armt` HEAD.

## Safe Branch Integration
- Before merging another branch into `armt`, run `git log --oneline --left-right --cherry-pick --graph armt...<branch>` and `git diff --name-status --find-renames armt...<branch>` to estimate divergence and overlap.
- If `git merge-base armt <branch>` already equals the current `armt` HEAD, use `git merge --ff-only <branch>` instead of a normal merge. This repo already used that fast-forward path when `gdn-new` was integrated into `armt`.

## Worktree Cleanup Order
- If a branch is checked out in a sibling worktree, remove the worktree first and delete the branch second. `git branch -d <branch>` is blocked while another worktree still owns that checkout.
- For intentionally disposable sibling worktrees, `git worktree remove --force <path>` is acceptable when the only remaining local files are scratch artifacts under `codex_assets/`.

## ARMT Test-Double Invariants
- `ARMTLayer` now exposes a single backend slot, `recurrent_memory_layer`, for associative, gated-deltanet, and cross-attention-slot memory backends.
- Any test that monkeypatches `ARMTLayer.__init__` must initialize `recurrent_memory_layer` to `None` before injecting mocks. Otherwise `_get_memory_layer()` can raise `AttributeError` from the test double even when production code is correct.
- For chunk-control tests, initialize `_skip_read_memory_for_current_chunk` and `_current_chunk_is_first` in the test double as well, so `reset_memory()` and first-chunk propagation tests match real-layer invariants.
- For ARMT/RMT-scoped changes, default to targeted verification on the affected ARMT/RMT tests and the directly related training/plumbing tests. Do not routinely rerun unrelated baseline Megatron tests that are outside the ARMT/RMT change surface unless the edit clearly crosses those boundaries.

## ARMT Memory Parameter Reporting
- `megatron/training/training.py` does not know backend-specific memory modules. The printed memory-parameter summary is driven entirely by `model.get_memory_parameter_breakdown()`.
- For ARMT, new recurrent memory backends automatically show up in the training-side memory summary if they are attached on each layer as `recurrent_memory_layer` and expose their trainable tensors through standard `nn.Module.parameters()`. No training-side special-case hook is needed.
- `--recurrent-memory-lr` groups parameters purely by the `*.recurrent_memory_layer.*` state-dict prefix and intentionally leaves top-level `memory_embeddings` on the base LR schedule.
- ARMT now keeps the `modules:` table coarse-grained and prints runtime memory-state summaries separately before that table: `initial_slots` totals come from `CrossAttentionSlotMemory.initial_slots`; `W_mem` totals are single-sample (`batch=1`) logical memory-store sizes. For `AssociativeLayer`, use pre-DPFP width (`d_mem * head_dim` per head), not the expanded `d_key = 2 * nu * d_mem`, so the report stays comparable with other backends' memory storage.
- ARMT recurrent backend head-dim decoupling is now projection-based: external hidden/state width stays `hidden_size`, while backend multi-head projection width may use explicit `head_dim`. If associative parameter shapes change, keep `examples/armt/tools/convert_baseline_to_armt.py` in sync with runtime shapes, especially `W_mv`, `W_mo`, and gated `W_mb`.

## ARMT Memory Write Sources
- `--armt-memory-write-source` is the source of truth for ARMT recurrent writes. `mem_tokens` keeps the original behavior; `post_mlp_context`, `post_attn_context`, and `pre_attn_context` all force `num_mem_tokens=0` during validation and therefore must also disable memory-token concat/slicing/monitoring paths at runtime.
- When `num_mem_tokens=0`, do not keep a zero-sized `ARMTModel.memory_embeddings` parameter around. DistOpt checkpoint save can then build a padding-only bucket and fail with `AssertionError: empty bucket encountered` in `sharded_param_state_dp_reshardable`.
- `pre_attn_context` is defined as the hidden state after memory read injection and immediately before attention. When `--recurrent-memory-input-pre-norm` is enabled in that mode, reuse `TransformerLayer.input_layernorm` output instead of instantiating or reapplying a backend-local recurrent pre-norm; otherwise the write path silently diverges from the real attention input and adds dead recurrent-norm parameters.

## ARMT GDN Backend Parity
- `GatedDeltaNetMemory` only matches native `megatron/core/ssm/gated_delta_net.py` projection init scales if `ARMTLayer` threads config-derived init metadata into the backend. If that plumbing is removed, the backend falls back to `num_layers=1` and silently over-initializes `out_proj`.
- Native-parity init for `GatedDeltaNetMemory` now depends on passing the real transformer `config` object into the backend. Keep `ARMTLayer` handing through the same config instance so `reset_parameters()` reads `params_dtype`, `perform_initialization`, `init_method`, and `output_layer_init_method` directly from the canonical source, matching native GDN and avoiding a second ad-hoc config namespace.
- Direct `GatedDeltaNetMemory(...)` construction now defaults `use_input_pre_norm=True`, but end-to-end ARMT runs still follow the explicit `recurrent_memory_input_pre_norm` flag passed by `ARMTLayer`. Do not assume the standalone constructor default reflects the training CLI default.
- `AssociativeLayer` and `CrossAttentionSlotMemory` now use the same pattern: direct construction requires the canonical `TransformerConfig`, and both runtime modules plus `examples/armt/tools/convert_baseline_to_armt.py` are expected to stay aligned on three Megatron init families (`embedding_init_method`, `init_method`, `output_layer_init_method`). If one side keeps an older special-case init (for example zero `W_mv` or fixed `initial_slots` std), checkpoint-conversion tests will drift from runtime behavior.

## ARMT TensorBoard Monitoring
- `ARMTModel.consume_all_monitoring_primitives()` always emits the aggregated `armt/...` metrics. Per-layer copies are emitted only when `--armt-log-layer-metrics-to-tensorboard` is enabled, and they now append the layer tag as `armt/.../layer_XX`; `XX` follows the transformer `layer_number` (or module order fallback in lightweight tests).
- Readback monitoring for all recurrent backends now uses a single denominator semantic again: `armt/read/retrieved_to_<partition>_hidden_ratio` compares the raw backend readout against the pre-injection hidden stream. In `residual` mode this is the addend magnitude; in gate modes it is the gate-input magnitude, not the final multiplicative delta. Optional `input_pre_norm` still changes the real read path, but it no longer emits a separate `..._memory_input_ratio` metric; the per-position companion metric is `armt/read/retrieved_to_hidden_ratio/pos_XXXX`.
- Chunk-scoped ARMT read metrics belong in `megatron/core/pipeline_parallel/recurrent_schedules.py`, not in `ARMTModel`: only the recurrent scheduler has the authoritative `chunk_idx`, and it can publish `armt/read/.../chunk_XX` while keeping the iteration-level aggregates merged from the same per-chunk primitives. When the first chunk skips memory reads, emit explicit zero-valued chunk metrics there instead of teaching each backend to invent its own chunk naming or zero-fill policy.
- `armt/read/retrieved_norm_mean/chunk_XX` and `armt/read/retrieved_to_hidden_ratio/chunk_XX` now have their own TensorBoard gate, `--armt-log-read-chunk-metrics-to-tensorboard`. Keep that switch enforced in the recurrent scheduler (where the chunk tags are created) and mirrored by a final training-side filter so pre-populated trackers or future call sites cannot leak chunk tags when the flag is off.
- ARMT-specific monitoring collection now follows `tensorboard_log_interval` itself, not just TensorBoard emission. On non-sampled steps, recurrent schedule publishing plus ARMT read/token/write/state reductions are skipped entirely, so future heavy monitoring should hook into the same per-iteration gate instead of accumulating every step and relying on later `clear_armt_tensorboard_metrics()`.
- For sampled steps, all DP ranks still participate in ARMT metric reduction, but only the actual TensorBoard writer rank should finalize the reduced primitives into Python metric dicts; non-writer ranks can return early after the collective. On the writer rank, emitting the ARMT scalar set as one batched TensorBoard `Summary` is measurably faster than thousands of per-tag `add_scalar()` calls and keeps the visible tags unchanged.
- Read-injection effect metrics (`armt/read/injection_*`) are emitted only for gate modes, not `residual`, and remain iteration/layer scoped only: do not add them to the chunk or position metric allowlists unless chunk/position semantics are explicitly requested.
- `--test-train-run` is the wrong benchmark mode for TensorBoard overhead or scalar validation in this repo: its no-artifact overrides drop `tensorboard_dir`, which disables the SummaryWriter path and skips the ARMT TensorBoard metric collection/consumption you would otherwise be timing. For a one-iteration TensorBoard smoke, omit `--test-train-run`, keep artifacts under `codex_assets/`, and remove the checkpoint directory afterward because exit-triggered checkpointing still saves at iteration 1.
- Keep `train/chunk_XX_loss` chunk-scoped only. It is published by the recurrent scheduler, not by individual ARMT layers, so do not treat it as a candidate for per-layer expansion.
- `train/avg_loss` is the recurrent scheduler's full-sequence token-weighted loss report. When `--no-loss-from-first-chunk` masks backward loss, this metric and `train/chunk_00_loss` still use the original first-chunk loss mask for logging; TensorBoard intentionally omits `train/avg_loss vs samples`.

## ARMT Read Injection Modes
- ARMT read injection currently exposes only `residual`, `silu_delta_gate`, and `sigmoid_gate`. Gate-style modes must be identity-centered for zero readouts because recurrent backends return zero before memory is written; avoid adding final gates whose zero-read behavior erases the residual stream.

## Recurrent CLI Source of Truth
- The canonical recurrent CLI flags live in `examples/recurrent/recurrent_args.py`, not in older ARMT scripts or memory.
- Prefer `--use-recurrent-model-schedule`, `--recurrent-chunk-size`, and `--recurrent-tbptt-mode` in new docs and launchers.
- `--use-armt-tbptt` and `--use-recurrent-tbptt` are intentionally rejected by `tests/unit_tests/models/armt/test_armt_constraints.py`; `--armt-chunk-size` survives only as a compatibility alias, not as the preferred spelling for new material.
- `--recurrent-mem-qk-norm` is the single ARMT memory qk-norm switch. It must drive `gated_deltanet`, `associative`, and `cross_attn_slots`; avoid reintroducing older backend-specific names such as `recurrent_slot_qk_norm`.
- `playground/rmt` launchers treat `ENABLE_RESUME=1` as best-effort: if `CHECKPOINT_PATH/latest_checkpointed_iteration.txt` is missing, they warn, set `ENABLE_RESUME=0`, and continue from scratch instead of failing before launch.
- Launcher env wiring mirrors the CLI name as `RECURRENT_MEM_QK_NORM`. Keep base launchers aligned with the code default (`false`), and put opt-in norm defaults only in explicit `*_w_norm.sh` variants together with `RECURRENT_MEMORY_INPUT_PRE_NORM=1`.
- The `--armt-d-mem`, `--armt-n-heads`, `--armt-head-dim`, `--armt-nu`, `--armt-use-denom`, `--armt-gating`, and `--armt-correction` knobs are associative-only. GDN and cross-attention-slot launchers should not wire them, and validation should not reject those backends because of associative divisibility rules.

## Profiling With No-Artifact Modes
- `--test-train-run` and `--param-stats-only` route through `megatron/training/arguments.py::_apply_no_artifact_run_overrides()`. If PyTorch profiler is enabled (`--profile --use-pytorch-profiler`), that override must preserve `tensorboard_dir`; the repo now reuses that path as the profiler trace directory for both TensorBoard-style exports and direct Perfetto JSON output.
- For profiler/smoke launchers, keep the real token-derived `TRAIN_ITERS` defaults intact and cap the short run with `EXIT_INTERVAL` instead. Overriding `TRAIN_ITERS` for no-artifact verification is now treated as the wrong pattern for this repo.
- Launcher-side `--tensorboard-dir` alone is not enough for this combination. When profiling smoke runs, verify the effective warning text mentions that `--tensorboard-dir` is being kept specifically as the PyTorch profiler trace directory.
- For no-artifact runs that keep `tensorboard_dir` only for native-profiler export, also suppress Megatron's `SummaryWriter`; otherwise the same directory silently accumulates `events.out.tfevents*` side files even though the intended artifact is just the Perfetto trace.
- The lightweight PyTorch-profiler path for this repo is now controlled by `--pytorch-profiler-record-shapes`, `--pytorch-profiler-with-stack`, `--pytorch-profiler-gzip-traces`, and `--pytorch-profiler-trace-format`. Keeping shapes/stacks disabled and using `perfetto` export preserves CPU op traces plus the GPU stream execution lanes (`kernel`, `gpu_memcpy`, `gpu_memset`) while dropping unrelated annotation/correlation payload from the final exported trace.

## ARMT Full Attention
- ARMT no longer owns a custom decoupled windowed full-attention path. Full attention now stays on the base GPT layer spec's self-attention; recurrent chunking only controls the recurrent schedule and memory read/write state.
- The removed implementation is preserved on the remote `decoupled` branch. Do not reintroduce `--full-attn-window-size`, `--armt-windowed-full-attn-backend`, `--armt-equal-window-full-attn-path`, or `ARMTSelfAttention` without first checking that branch for the old cache, rotary-offset, and launcher assumptions.
