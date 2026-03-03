# ARMT Example Tests (`examples/armt/tests`)

## Core Function
- 覆盖 ARMT 的“能跑通”集成路径（GPU forward/backward）与 checkpoint 转换工具行为。

## Directory Structure
- `test_armt_train.py`: GPU 集成冒烟（单卡/TP=2）；真实构建 `ARMTModel` 并做一次反传。
- `test_armt_checkpoint.py`: CPU 测试（baseline->ARMT 转换后 state_dict key/shape 断言）。

## Key Data Flow
- `test_armt_train.py`
  1. `dist.init_process_group("nccl")`
  2. `initialize_model_parallel(TP, PP=1, CP=1)`
  3. 构建小模型 `ARMTModel(get_armt_layer_spec(...))`
  4. 构造 batch（含 `attention_mask/position_ids/loss_mask`）并 `loss.backward()`
- `test_armt_checkpoint.py`
  - 调用 `examples/armt/tools/convert_baseline_to_armt.py` 并验证新增参数/不保存 runtime buffers。

## Dev Notes
- `test_armt_train.py` 依赖 CUDA/NCCL；无 GPU 会自动 skip。
- 修改 ARMT 参数命名/shape 时优先更新：
  - `examples/armt/tools/convert_baseline_to_armt.py`
  - `examples/armt/tests/test_armt_checkpoint.py`

