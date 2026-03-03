# ARMT Config Samples (`examples/armt/configs`)

## Core Function
- 保存 ARMT 参数的 YAML 样例，便于在外部 runner/配置系统中复用。

## Directory Structure
- `armt_3b_example.yaml`: 最小字段集合示例（与 CLI 参数同名/近似）。

## Key Data Flow
- 该目录 YAML **不会**被 `examples/armt/train.py` 自动读取（入口使用 argparse）。
- 典型使用方式：外部脚本/平台读取 YAML 后映射为 CLI flags（例如 `use_armt_tbptt` -> `--use-armt-tbptt`）。

## Dev Notes
- 若新增/重命名 ARMT CLI 参数，请同步更新：
  - `examples/armt/armt_args.py`
  - 该目录 YAML 样例（保持字段对齐）

