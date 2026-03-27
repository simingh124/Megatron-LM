# PROGRAM

## 2026-03-27 gdn-new -> armt merge note
- Merge target was `armt`; source branch was `gdn-new`.
- Before merging, `git merge-base armt gdn-new` returned `9d91624d9eaffe4e9d63d9601783115c82d6e9ca`, which matched the then-current `armt` HEAD. This means the integration path was a fast-forward, not a real three-way merge.
- The safe command for this case is `git merge --ff-only gdn-new`. It advanced `armt` to `88b21fc8f99acb0a49fdfdd883ae304735a0fd95` without conflicts.

## Reusable conflict precheck
- Run `git log --oneline --left-right --cherry-pick --graph armt...<branch>` first to see whether both branches have diverged.
- Run `git diff --name-status --find-renames armt...<branch>` to estimate the file overlap before touching the worktree.
- If `merge-base` already equals `armt` HEAD, prefer `--ff-only`. It fails fast instead of dropping the repo into a half-merged state.

## Implicit repo constraints observed
- `gdn-new` is already checked out in sibling worktree `/mnt/step3-abla/siming/code_repo/Megatron-LM-gdn-new`. Merge it from the `armt` worktree directly; do not try to re-check out `gdn-new` in the root worktree.
- Root worktree currently contains user-side untracked assets under `codex_assets/`. Leave them untouched unless the task explicitly targets them.
- The repo-level process requires `PROGRAM.md` to exist and be read before development actions. Keep this file updated after every code-changing task so later work can reuse the same pitfalls and decision trail.

## 2026-03-27 AGENTS / PROGRAM split note
- Stable, always-on execution rules belong in `AGENTS.md` so they are visible before work starts. Keep `PROGRAM.md` for transient discoveries such as branch topology, failed paths, hidden dependencies, and review pitfalls.
- When summarizing "current changes", run `git status --short` before relying on `git diff --stat`. Purely untracked files do not appear in diff-only summaries.
- If `codex_assets/` contains explanatory diagrams or exports, either pair them with the narrative doc in the same commit or leave them out of the commit. Orphaned assets are hard to review later.
