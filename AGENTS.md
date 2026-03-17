# Repository Guidelines

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
- `pip install --no-build-isolation .[mlm,dev]`: install local package and Megatron-LM extras.
- `uv sync --locked --only-group linting` and `uv sync --only-group test`: reproduce CI lint/test environments.
- `bash tools/autoformat.sh`: run Black, isort, pylint, ruff, and mypy (best-effort) on changed files.
- `uv run pytest tests/unit_tests/models/armt -v`: ARMT unit tests.
- `uv run pytest examples/armt/tests/test_armt_train.py -v`: ARMT GPU integration smoke test.
- `python examples/armt/train.py --use-armt-tbptt --num-mem-tokens 16 --armt-chunk-size 512 ...`: local ARMT training entrypoint.

## Coding Style & Naming Conventions
- Python style uses 4-space indentation and max line length `100`.
- Formatting/linting follows repo config: Black `24.4.2`, isort (Black profile), ruff, pylint, flake8.
- Prefer `snake_case` for functions/files, `PascalCase` for classes, and explicit typing on new/changed interfaces.
- Avoid `print` in core paths (`pylint` flags it); keep edits scoped and avoid unrelated reformatting.

## ARMT-Specific Guardrails
- Validate constraints before large runs: PP must be `1`, CP must be `1`, FP8 unsupported, and position embedding must be `rope` or `yarn`.
- With sequence parallel enabled, ensure `(armt_chunk_size + num_mem_tokens) % TP == 0`.
- TBPTT chunking behavior is implemented in `megatron/core/pipeline_parallel/recurrent_schedules.py`; update tests when changing chunk semantics.

## Testing Guidelines
- Test framework is `pytest` (`python_files = test_*.py`).
- Useful markers include `internal`, `flaky`, and `flaky_in_dev`.
- Keep tests near the feature: ARMT logic in `tests/unit_tests/models/armt/`, runtime integration in `examples/armt/tests/`.
- For ARMT changes, minimum gate is:
  - unit tests covering model/layer/constraints touched;
  - one GPU forward/backward smoke test;
  - checkpoint-conversion test when state format or memory params change.

## Commit & Pull Request Guidelines
- Follow current history style: `fix:`, `chore:`, `build:`, `ci:` + imperative summary (optionally `(#PR)`).
- Keep commits atomic and scoped; avoid unrelated formatting churn.
- PRs should complete template checks: unit tests, functional tests (if applicable), typing, docs, and `tools/autoformat.sh`.
- Apply labels in order: `Expert Review` first, then `Final Review` after approvals and green CI.
