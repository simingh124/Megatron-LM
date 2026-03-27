# Repository Guidelines

## First Read
- Read `PROGRAM.md` before any project modification, debugging, benchmark run, or branch integration. It is the project memory for trial-and-error paths, implicit dependencies, worktree topology, and recurring pitfalls.
- Keep stable repo rules in `AGENTS.md`; keep task-specific discoveries and pitfalls in `PROGRAM.md`. After every project change, append concise, accurate notes to `PROGRAM.md` and remove stale guidance when it is no longer true.
- Unless the user explicitly requests another location, save Codex-generated artifacts under `codex_assets/`. Do not touch unrelated user assets in that directory.
- Before summarizing workspace changes or proposing a commit split, run `git status --short` first. Untracked files do not appear in `git diff --stat`.

## Project Operating Rules
- Network proxy: before any terminal command that needs network access, run `eval $(curl -s http://deploy.i.shaipower.com/httpproxy)`. If connectivity still fails, run `unset https_proxy http_proxy all_proxy` and retry once without the proxy.
- `brainctl` path is fixed: always call `/kubebrain/brainctl` directly. Do not use `which` or `find` to discover it.
- Python and `pip` must stay inside `/mnt/step3-abla/siming/.venv/`. Prefer absolute paths such as `/mnt/step3-abla/siming/.venv/bin/python` and `/mnt/step3-abla/siming/.venv/bin/pip`.
- `pip` install priority:
  1. `/mnt/step3-abla/siming/.venv/bin/pip install -i https://artifactory.stepfun-inc.com/artifactory/api/pypi/pypi-public/simple/ --trusted-host artifactory.stepfun-inc.com <package>`
  2. `/mnt/step3-abla/siming/.venv/bin/pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ <package>`
  3. `/mnt/step3-abla/siming/.venv/bin/pip install -i https://pypi.org/simple <package>`
- Fail fast: avoid broad `try...except` unless there is a concrete recovery path that the caller requires. Prefer surfacing exceptions directly.
- Do not guess. If requirements, context, or a decision boundary are unclear and cannot be verified locally, stop and ask the user instead of inventing behavior.
- Before any task that needs 8 GPUs, release the placeholder load by running `python /home/i-huangsiming/work/tools/gpu_util.py` and terminating the occupying script as needed. Restart the same script after the task completes so the reservation is restored.
- Megatron-LM training smoke tests must default to `--test-train-run`. Unless the user explicitly asks for logs, TensorBoard, or checkpoints, do not create persistent training artifacts.

## Project Structure & Module Organization
- Core runtime and model code: `megatron/core/` (parallelism, transformer blocks, optimizers, datasets, inference).
- Training infrastructure and argument plumbing: `megatron/training/`; reference entrypoints at repo root (`pretrain_gpt.py`, `pretrain_t5.py`, `train_rl.py`).
- ARMT focus paths:
  - `megatron/core/models/armt/`
  - `megatron/core/pipeline_parallel/recurrent_schedules.py`
  - `examples/armt/` (entrypoint, configs, conversion tool)
  - `tests/unit_tests/models/armt/` and `examples/armt/tests/`
- Testing split: `tests/unit_tests/` (fast), `tests/functional_tests/test_cases/` (YAML-driven integration), `tests/test_utils/` (shared launch tooling).

## Build, Test, and Development Commands
- `/mnt/step3-abla/siming/.venv/bin/pip install --no-build-isolation .[mlm,dev]`: install the local package and Megatron-LM extras into the mandated venv.
- `uv sync --locked --only-group linting` and `uv sync --only-group test`: reproduce CI lint/test environments when parity with CI matters more than local iteration speed.
- `bash tools/autoformat.sh`: run Black, isort, pylint, ruff, and mypy (best-effort) on changed files.
- `/mnt/step3-abla/siming/.venv/bin/python -m pytest tests/unit_tests/models/armt -v`: ARMT unit tests.
- `/mnt/step3-abla/siming/.venv/bin/python -m pytest examples/armt/tests/test_armt_train.py -v`: ARMT GPU integration smoke test.
- `/mnt/step3-abla/siming/.venv/bin/python examples/armt/train.py --use-recurrent-model-schedule --recurrent-chunk-size 512 --recurrent-tbptt-mode --num-mem-tokens 16 --test-train-run ...`: local ARMT training entrypoint with the required non-persistent test mode and canonical recurrent flags.

## Coding Style & Naming Conventions
- Python style uses 4-space indentation and max line length `100`.
- Formatting and linting follow repo config: Black `24.4.2`, isort (Black profile), ruff, pylint, and flake8.
- Prefer `snake_case` for functions/files, `PascalCase` for classes, and explicit typing on new or changed interfaces.
- Avoid `print` in core paths (`pylint` flags it); keep edits scoped and avoid unrelated reformatting.

## ARMT-Specific Guardrails
- Validate constraints before large runs: PP must be `1`, CP must be `1`, FP8 is unsupported, and position embedding must be `rope` or `yarn`.
- With sequence parallel enabled, ensure `(armt_chunk_size + num_mem_tokens) % TP == 0`.
- TBPTT chunking behavior is implemented in `megatron/core/pipeline_parallel/recurrent_schedules.py`; update tests whenever chunk semantics change.
- Prefer canonical recurrent CLI flags in new docs/scripts: `--use-recurrent-model-schedule`, `--recurrent-chunk-size`, and `--recurrent-tbptt-mode`. The legacy toggles `--use-armt-tbptt` and `--use-recurrent-tbptt` are intentionally rejected by argument parsing tests.

## Testing Guidelines
- Test framework is `pytest` (`python_files = test_*.py`).
- Useful markers include `internal`, `flaky`, and `flaky_in_dev`.
- Keep tests near the feature: ARMT logic in `tests/unit_tests/models/armt/`, runtime integration in `examples/armt/tests/`.
- When launching training benchmarks or test runs with overridden `TRAIN_TOKENS`, set it high enough that the derived `lr_warmup_steps` stays strictly smaller than `lr_decay_steps`; avoid configurations where warmup is greater than or equal to decay.
- For ARMT changes, the minimum verification gate is:
  - unit tests covering the model, layer, or constraints touched;
  - one GPU forward/backward smoke test;
  - checkpoint-conversion coverage when state format or memory parameters change.

## Commit & Pull Request Guidelines
- Follow current history style: `feat:`, `fix:`, `docs:`, `refactor:`, `chore:`, `build:`, `ci:` plus an imperative summary (optionally `(#PR)`).
- Keep commits atomic and scoped. Split process/docs changes from model or experiment artifacts when they are independently reviewable.
- PRs should complete template checks: unit tests, functional tests (if applicable), typing, docs, and `tools/autoformat.sh`.
- Apply labels in order: `Expert Review` first, then `Final Review` after approvals and green CI.
