# PROGRAM

## Purpose
- This file stores repo-specific operational knowledge that is easy to forget and expensive to rediscover: branch/worktree topology, hidden coupling, test-double invariants, and doc/code drift traps.
- Stable always-on rules belong in `AGENTS.md`; do not duplicate them here unless the repo has a concrete exception or counterexample.

## Branch and Worktree Topology
- `armt` is the integration branch in the root worktree `/mnt/step3-abla/siming/code_repo/Megatron-LM`.
- `cross` lives in sibling worktree `/mnt/step3-abla/siming/code_repo/Megatron-LM-cross` and currently starts from `armt` commit `5e36844ec`. If it needs to be recreated from a newer `armt`, remove the sibling worktree first and then create the branch/worktree explicitly from the refreshed `armt` HEAD.
- `overlap` lives in sibling worktree `/mnt/step3-abla/siming/code_repo/Megatron-LM-overlap` and currently starts from `armt` commit `5e36844ec`. If it needs to be recreated from a newer `armt`, remove the sibling worktree first and then create the branch/worktree explicitly from the refreshed `armt` HEAD.

## Safe Branch Integration
- Before merging another branch into `armt`, run `git log --oneline --left-right --cherry-pick --graph armt...<branch>` and `git diff --name-status --find-renames armt...<branch>` to estimate divergence and overlap.
- If `git merge-base armt <branch>` already equals the current `armt` HEAD, use `git merge --ff-only <branch>` instead of a normal merge. This repo already used that fast-forward path when `gdn-new` was integrated into `armt`.

## Worktree Cleanup Order
- If a branch is checked out in a sibling worktree, remove the worktree first and delete the branch second. `git branch -d <branch>` is blocked while another worktree still owns that checkout.
- For intentionally disposable sibling worktrees, `git worktree remove --force <path>` is acceptable when the only remaining local files are scratch artifacts under `codex_assets/`.

## ARMT Test-Double Invariants
- Since commit `5e491de47`, `ARMTLayer` no longer assumes a single associative backend. It exposes two backend slots, `associative_layer` and `recurrent_memory_layer`, and resolves them through `_get_memory_layer()`.
- Any test that monkeypatches `ARMTLayer.__init__` must initialize both backend slots to `None` before injecting mocks. Otherwise `_get_memory_layer()` can raise `AttributeError` from the test double even when production code is correct.
- For chunk-control tests, initialize `_skip_read_memory_for_current_chunk` and `_current_chunk_is_first` in the test double as well, so `reset_memory()` and first-chunk propagation tests match real-layer invariants.
- For ARMT/RMT-scoped changes, default to targeted verification on the affected ARMT/RMT tests and the directly related training/plumbing tests. Do not routinely rerun unrelated baseline Megatron tests that are outside the ARMT/RMT change surface unless the edit clearly crosses those boundaries.

## Recurrent CLI Source of Truth
- The canonical recurrent CLI flags live in `examples/recurrent/recurrent_args.py`, not in older ARMT scripts or memory.
- Prefer `--use-recurrent-model-schedule`, `--recurrent-chunk-size`, and `--recurrent-tbptt-mode` in new docs and launchers.
- `--use-armt-tbptt` and `--use-recurrent-tbptt` are intentionally rejected by `tests/unit_tests/models/armt/test_armt_constraints.py`; `--armt-chunk-size` survives only as a compatibility alias, not as the preferred spelling for new material.

## Profiling With No-Artifact Modes
- `--test-train-run` and `--param-stats-only` route through `megatron/training/arguments.py::_apply_no_artifact_run_overrides()`. If PyTorch profiler is enabled (`--profile --use-pytorch-profiler`), that override must preserve `tensorboard_dir`; the repo now reuses that path as the profiler trace directory for both TensorBoard-style exports and direct Perfetto JSON output.
- For profiler/smoke launchers, keep the real token-derived `TRAIN_ITERS` defaults intact and cap the short run with `EXIT_INTERVAL` instead. Overriding `TRAIN_ITERS` for no-artifact verification is now treated as the wrong pattern for this repo.
- Launcher-side `--tensorboard-dir` alone is not enough for this combination. When profiling smoke runs, verify the effective warning text mentions that `--tensorboard-dir` is being kept specifically as the PyTorch profiler trace directory.
- For no-artifact runs that keep `tensorboard_dir` only for native-profiler export, also suppress Megatron's `SummaryWriter`; otherwise the same directory silently accumulates `events.out.tfevents*` side files even though the intended artifact is just the Perfetto trace.
- The lightweight PyTorch-profiler path for this repo is now controlled by `--pytorch-profiler-record-shapes`, `--pytorch-profiler-with-stack`, `--pytorch-profiler-gzip-traces`, and `--pytorch-profiler-trace-format`. Keeping shapes/stacks disabled and using `perfetto` export preserves CPU op traces plus the GPU stream execution lanes (`kernel`, `gpu_memcpy`, `gpu_memset`) while dropping unrelated annotation/correlation payload from the final exported trace.

## ARMT Windowed Full Attention
- Decoupled ARMT full attention is implemented in `megatron/core/models/armt/armt_self_attention.py`, not by changing Megatron's global sliding-window config. The layer caches only historical real-token K/V; historical memory-token K/V is intentionally excluded.
- `full_attn_window_size == recurrent_chunk_size` is a deliberate legacy fast path. In that setting ARMT must fall back to the old non-overlap attention path to keep 10-step losses exactly aligned with the pre-change implementation.
- Decoupled windows currently only support no-TBPTT and reject `sequence_parallel`; the validation lives in `examples/recurrent/recurrent_args.py`, so launchers should rely on that source of truth instead of re-encoding the rule locally.
- The `playground/rmt/qwen3_0p6b_armt_gdn_0324_fs_wo_tbptt_nmem64_swa512_chunk64.sh` launcher must pass `--full-attn-window-size` explicitly; defining `FULL_ATTN_WINDOW_SIZE` in the shell is not enough and will silently run the legacy equal-window path if the CLI flag is omitted.
- For the same launcher family, `micro-batch-size=1` with `global-batch-size=480` creates 60 gradient-accumulation microsteps on 8 GPUs and makes the first logged iteration look hung. Use the validated windowed defaults (`mb=4`, `gb=32`) or direct `TRAIN_ITERS/LR_*_ITERS` overrides for smoke checks.
- For `playground/rmt/qwen3_0p6b_armt_gdn_0324_fs_wo_tbptt_nmem64_swa512_chunk128.sh`, the old `micro-batch-size=5` default was the main throughput bottleneck on 8 GPUs: in a short `--test-train-run` benchmark it averaged about `6.47 TFLOP/s/GPU`, while keeping the same `global-batch-size=480` and raising only `micro-batch-size` to `10` raised the same benchmark to about `13.45 TFLOP/s/GPU` with matching early-step losses and 24-GPU divisibility preserved. Lowering `NUM_WORKERS` from `32` to `16` in the same setup did not show a material extra gain, so treat the micro-batch default as the proven knob and avoid churning worker-count defaults without fresh measurements.
- The `chunk128 + swa512` hot path benefits materially from the SDPA-based windowed attention path plus chunk-queue history caching in `megatron/core/models/armt/armt_self_attention.py`: on the same 8-GPU smoke benchmark, keeping `micro-batch-size=5` and changing only that hot path improved steady-state throughput from about `6.47` to `6.78 TFLOP/s/GPU` while keeping the first several losses aligned within normal BF16 drift.
- The ARMT decoupled windowed full-attention path cannot use PyTorch SDPA's flash backend as a drop-in because that backend rejects the current `seqlen_q != seqlen_k` rectangular causal layout. An explicit `flash_attn` branch in `megatron/core/models/armt/armt_self_attention.py` does work for this path, but on the `qwen3_0p6b_armt_gdn_0324_fs_wo_tbptt_nmem64_swa512_chunk128_test.sh` 8-GPU smoke run it did not show a steady-state throughput gain over the native backend and the later 10-step losses drifted more than the early-step BF16 noise floor, so keep `ARMT_WINDOWED_FULL_ATTN_BACKEND` defaulted to `native` and treat `flash_attn` as an explicit experimental backend only.
- Equal-window ARMT full attention is no longer hard-wired to the legacy TE path: `--armt-equal-window-full-attn-path {legacy,window}` now controls whether `full_attn_window_size == recurrent_chunk_size` stays on the old TE fast path or is forced onto the custom window-attention path. Keep the default at `legacy` for backward-compatible loss alignment.
- On the 8-GPU `qwen3_0p6b_armt_gdn_0324_fs_wo_tbptt_nmem64` family, forcing `chunk512 + swa512` onto the window-attention path via `qwen3_0p6b_armt_gdn_0324_fs_wo_tbptt_nmem64_swa512_chunk512.sh` did not reproduce the severe `chunk256 + swa512` slowdown: steady-state throughput stayed around `59.5 TFLOP/s/GPU` versus about `57.3` for the legacy TE baseline, while `chunk256 + swa512` stayed near `26.8`. Treat chunk fragmentation and degraded overlap from doubling the recurrent chunks as the dominant performance tax; the TE-vs-window attention implementation alone is not the main bottleneck in this launcher family.
