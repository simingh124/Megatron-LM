# PROGRAM

## Purpose
- This file stores repo-specific operational knowledge that is easy to forget and expensive to rediscover: branch/worktree topology, hidden coupling, test-double invariants, and doc/code drift traps.
- Stable always-on rules belong in `AGENTS.md`; do not duplicate them here unless the repo has a concrete exception or counterexample.

## Branch and Worktree Topology
- `armt` is the integration branch in the root worktree `/mnt/step3-abla/siming/code_repo/Megatron-LM`.
- `gdn-new` already lives in sibling worktree `/mnt/step3-abla/siming/code_repo/Megatron-LM-gdn-new`. Merge it from the `armt` worktree; do not try to re-check out `gdn-new` in the root worktree.
- `param` already lives in sibling worktree `/mnt/step3-abla/siming/code_repo/Megatron-LM-param`. If it needs to start from the latest `armt`, push `armt` first and create/update it explicitly from `armt`.

## Safe Branch Integration
- Before merging another branch into `armt`, run `git log --oneline --left-right --cherry-pick --graph armt...<branch>` and `git diff --name-status --find-renames armt...<branch>` to estimate divergence and overlap.
- If `git merge-base armt <branch>` already equals the current `armt` HEAD, use `git merge --ff-only <branch>` instead of a normal merge. This repo has already hit that fast-forward case with `gdn-new`.

## ARMT Test-Double Invariants
- Since commit `5e491de47`, `ARMTLayer` no longer assumes a single associative backend. It exposes two backend slots, `associative_layer` and `recurrent_memory_layer`, and resolves them through `_get_memory_layer()`.
- Any test that monkeypatches `ARMTLayer.__init__` must initialize both backend slots to `None` before injecting mocks. Otherwise `_get_memory_layer()` can raise `AttributeError` from the test double even when production code is correct.
- For chunk-control tests, initialize `_skip_read_memory_for_current_chunk` and `_current_chunk_is_first` in the test double as well, so `reset_memory()` and first-chunk propagation tests match real-layer invariants.
- For ARMT/RMT-scoped changes, default to targeted verification on the affected ARMT/RMT tests and the directly related training/plumbing tests. Do not routinely rerun unrelated baseline Megatron tests that are outside the ARMT/RMT change surface unless the edit clearly crosses those boundaries.

## Recurrent CLI Source of Truth
- The canonical recurrent CLI flags live in `examples/recurrent/recurrent_args.py`, not in older ARMT scripts or memory.
- Prefer `--use-recurrent-model-schedule`, `--recurrent-chunk-size`, and `--recurrent-tbptt-mode` in new docs and launchers.
- `--use-armt-tbptt` and `--use-recurrent-tbptt` are intentionally rejected by `tests/unit_tests/models/armt/test_armt_constraints.py`; `--armt-chunk-size` survives only as a compatibility alias, not as the preferred spelling for new material.
