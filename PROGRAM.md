# PROGRAM

## Purpose
- This file stores repo-specific operational knowledge that is easy to forget and expensive to rediscover: branch/worktree topology, hidden coupling, test-double invariants, and doc/code drift traps.
- Stable always-on rules belong in `AGENTS.md`; do not duplicate them here unless the repo has a concrete exception or counterexample.

## Branch and Worktree Topology
- `armt` is the integration branch in the root worktree `/mnt/step3-abla/siming/code_repo/Megatron-LM`.
- `param` already lives in sibling worktree `/mnt/step3-abla/siming/code_repo/Megatron-LM-param`. If it needs to start from the latest `armt`, push `armt` first and create/update it explicitly from `armt`.

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

## ARMT Memory Parameter Reporting
- `megatron/training/training.py` does not know backend-specific memory modules. The printed memory-parameter summary is driven entirely by `model.get_memory_parameter_breakdown()`.
- For ARMT, new recurrent memory backends automatically show up in the training-side memory summary if they are attached on each layer as `recurrent_memory_layer` and expose their trainable tensors through standard `nn.Module.parameters()`. No training-side special-case hook is needed.
- ARMT now keeps the `modules:` table coarse-grained and prints runtime memory-state summaries separately before that table: `initial_slots` totals come from `CrossAttentionSlotMemory.initial_slots`; `W_mem` totals are single-sample (`batch=1`) logical memory-store sizes. For `AssociativeLayer`, use pre-DPFP width (`d_mem * head_dim` per head), not the expanded `d_key = 2 * nu * d_mem`, so the report stays comparable with other backends' memory storage.
- ARMT recurrent backend head-dim decoupling is now projection-based: external hidden/state width stays `hidden_size`, while backend multi-head projection width may use explicit `head_dim`. If associative parameter shapes change, keep `examples/armt/tools/convert_baseline_to_armt.py` in sync with runtime shapes, especially `W_mv`, `W_mo`, and gated `W_mb`.

## Recurrent CLI Source of Truth
- The canonical recurrent CLI flags live in `examples/recurrent/recurrent_args.py`, not in older ARMT scripts or memory.
- Prefer `--use-recurrent-model-schedule`, `--recurrent-chunk-size`, and `--recurrent-tbptt-mode` in new docs and launchers.
- `--use-armt-tbptt` and `--use-recurrent-tbptt` are intentionally rejected by `tests/unit_tests/models/armt/test_armt_constraints.py`; `--armt-chunk-size` survives only as a compatibility alias, not as the preferred spelling for new material.
- `--recurrent-mem-qk-norm` is the single ARMT memory qk-norm switch. It must drive `gated_deltanet`, `associative`, and `cross_attn_slots`; avoid reintroducing older backend-specific names such as `recurrent_slot_qk_norm`.
- Launcher env wiring mirrors the CLI name as `RECURRENT_MEM_QK_NORM`. Keep base launchers aligned with the code default (`false`), and put opt-in norm defaults only in explicit `*_w_norm.sh` variants together with `RECURRENT_MEMORY_INPUT_PRE_NORM=1`.
