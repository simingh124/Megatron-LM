# ARMT Unit Tests (`tests/unit_tests/models/armt`)

## Core Function
- ARMT v1 的快速单测（CPU 为主），覆盖：约束检查、模型/层行为、TBPTT chunk 调度、associative memory 组件。

## Directory Structure
- `test_armt_constraints.py`: `validate_armt_constraints()` 的硬约束。
- `test_armt_model.py`: `ARMTModel` 的 memory concat/strip 与 RoPE 扩展。
- `test_armt_layer.py`: `ARMTLayer` 的调用顺序与 memory token 写入 slice。
- `test_armt_scheduler.py`: `chunk_data` 与 `--no-loss-from-first-chunk` 语义（含 skip-backward）。
- `test_associative_layer.py`: `DPFP/AssociativeLayer` 的 shape、reset、TBPTT detach、memory 累积更新。
- `conftest.py`: 覆盖全局测试数据下载逻辑（保持单测纯净）。

## Key Data Flow
- 单测多用 mock/patch 替换 Megatron 重型初始化，聚焦：
  - shape/参数命名契约
  - schedule 对 batch key 的切片与错误处理
  - TBPTT detach 断言（buffer 无 `grad_fn`）

## Dev Notes
- 推荐运行：`uv run pytest tests/unit_tests/models/armt -v`
- 若变更：
  - 约束（PP/CP/FP8/SP 整除）=> 更新 `test_armt_constraints.py`
  - memory token 拼接/attention mask 规则 => 更新 `test_armt_model.py`
  - chunk 切分/首 chunk loss 行为 => 更新 `test_armt_scheduler.py`

