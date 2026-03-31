# Playground Launcher Writing Guide

适用范围：
- `playground/rmt/*.sh`
- `playground/*.sh` 中沿用同一风格的训练启动脚本

目标：
- 统一训练脚本的变量组织方式，降低对比不同实验脚本时的阅读成本。
- 保持“只看变量定义区就能理解实验差异”的可读性。
- 在不改变训练逻辑的前提下，让 ARMT、RMT、baseline、MLM 脚本尽量共享同一套排版规范。

## 一、固定分段顺序

变量定义区统一按下面顺序书写，并使用完全一致的分隔线：

1. `# ========== Communication / runtime control ==========`
2. `# ========== Paths (files and data) ==========`
3. `# ========== Model structure parameters ==========`
4. `# ========== ARMT mechanism parameters ==========` 或 `# ========== RMT mechanism parameters ==========`  
   baseline / MLM 脚本没有机制参数段时可省略。
5. `# ========== Training parameters ==========`

约束：
- 不再额外使用 `For test`、`Distributed runtime toggles`、`Distributed training setup`、`Fixed paths`、`Data`、`Fixed model parameters`、`ARMT parameters`、`RMT parameters`、`Fixed training parameters` 这类旧分隔线。
- 除这 5 个主分段外，其他说明统一使用普通注释，不再使用 `# ========== ... ==========` 形式。

## 二、各分段应放什么

### 1. Communication / runtime control

放“脚本如何运行”的控制变量，包括：
- `ENABLE_TEST_TRAIN_RUN`
- 分布式通信与运行时开关：
  `USE_DISTRIBUTED_OPTIMIZER`
  `OVERLAP_GRAD_REDUCE`
  `OVERLAP_PARAM_GATHER`
  `USE_NCCL_UB`
  `LOG_THROUGHPUT`
- 分布式拓扑：
  `GPUS_PER_NODE`
  `NUM_NODES`
  `NODE_RANK`
  `MASTER_ADDR`
  `MASTER_PORT`
  `WORLD_SIZE`
- 其他运行控制：
  `CUDA_DEVICE_MAX_CONNECTIONS`
  `EXIT_INTERVAL`
  `LOG_INTERVAL`

说明：
- `ENABLE_TEE_LOG` 的默认值推导如果依赖 `ENABLE_TEST_TRAIN_RUN`，推荐也放在这一段。
- 但真正依赖 `LOG_DIR` 的 tee 初始化代码块，可以保留在路径段后半部分。

### 2. Paths (files and data)

放所有路径类变量，以及与路径直接绑定的数据源配置，包括：
- 环境/仓库路径：
  `ROOT`
  `SCRIPT_DIR`
  `REPO_ROOT`
  `MEGATRON_ROOT`
  `VENV_PYTHON`
  `PYTHONPATH`
- 训练入口和资源路径：
  `PRETRAIN_SCRIPT_PATH`
  `LOAD_CHECKPOINT_PATH`
  `TOKENIZER_DIR`
  `CHECKPOINT_PATH`
  `TENSORBOARD_LOGS_PATH`
  `LOG_DIR`
  `DATA_CACHE_PATH`
- 数据路径：
  `DATASET_PATH`

这一段还允许放：
- `EXP_NAME` 这类用于拼 checkpoint / log 路径的脚本名派生变量
- `mkdir -p` 创建输出目录
- 路径存在性检查
- 依赖日志目录的 tee 初始化代码

说明：
- 数据路径归入这一段，不单独再起 `Data` 段。
- 如果 checkpoint 路径选择依赖 `TP_SIZE` 等模型变量，允许使用普通注释写一个“模型/checkpoint 选择”小块，但不要新增新的主分隔线。

### 3. Model structure parameters

放 backbone 结构参数，包括：
- 并行结构：
  `TP_SIZE`
  `PP_SIZE`
  `CP_SIZE`
- 层数与维度：
  `NUM_LAYERS`
  `HIDDEN_SIZE`
  `FFN_HIDDEN_SIZE`
  `NUM_ATTN_HEADS`
  `NUM_QUERY_GROUPS`
  `KV_CHANNELS`
- 序列与位置编码：
  `SEQ_LENGTH`
  `MAX_POSITION_EMBEDDINGS`
  `ROTARY_BASE`
  `ROTARY_PERCENT`
- 词表与归一化：
  `VOCAB_SIZE`
  `MAKE_VOCAB_SIZE_DIVISIBLE_BY`
  `NORM_EPS`
- 其他纯 backbone 结构参数，如：
  `BASELINE_VIRTUAL_CHUNK_SIZE`

要求：
- 这一段只放 backbone 结构，不放 ARMT/RMT 记忆机制开关。

### 4. ARMT / RMT mechanism parameters

ARMT 脚本放：
- `NUM_MEM_TOKENS`
- `ARMT_CHUNK_SIZE`
- `ARMT_N_HEADS`
- `ADD_NO_RECURRENT_TBPTT_MODE`
- `NO_READ_MEMORY_FROM_FIRST_CHUNK`

RMT 脚本放：
- `NUM_MEM_TOKENS`
- `RECURRENT_CHUNK_SIZE`
- `ADD_NO_RECURRENT_TBPTT_MODE`
- `NO_READ_MEMORY_FROM_FIRST_CHUNK`

要求：
- 这一段只放 recurrent/association 相关控制变量。
- 与机制相关的 CLI 组装继续使用原脚本已有数组命名，如 `ARMT_ARGS`、`RECURRENT_ARGS`。

### 5. Training parameters

这一段统一分成两部分，中间必须空一行：

第一部分：直接训练控制变量
- `MICRO_BATCH_SIZE`
- `GLOBAL_BATCH_SIZE`
- `NUM_WORKERS`
- `TRAIN_TOKENS`
- `LR_DECAY_TOKENS`
- `LR`
- `MIN_LR`

第二部分：由训练参数派生出来的变量
- `WARMUP_TOKENS`
- `TRAIN_ITERS`
- `LR_WARMUP_ITERS`
- `LR_DECAY_ITERS`

要求：
- 派生变量必须继续放在训练段内，不单独拆分到别处。
- 直接控制变量和派生变量之间必须保留一个空行。
- 与训练参数一致性相关的校验逻辑也应紧跟在这一段后面，例如：
  `LR_WARMUP_ITERS < LR_DECAY_ITERS`
  `OVERLAP_PARAM_GATHER` 对其他开关的依赖检查

## 三、主分段之后的内容

变量定义区结束后，继续保持下面顺序：

1. `DISTRIBUTED_ARGS`
2. `MODEL_ARGS`
3. `ARMT_ARGS` / `RECURRENT_ARGS`
4. `TRAINING_ARGS`
5. `DATA_ARGS`
6. `CKPT_AND_LOG_ARGS`
7. `EXTRA_ARGS`
8. `echo` 关键参数摘要
9. 最终 `torchrun` / `python -m torch.distributed.run` 启动命令

说明：
- 这些数组是命令行组装结果，不属于前面的“控制变量分段”。
- 这里不强制新增主分隔线，避免脚本下半部分被切得过碎。

## 四、注释与默认值规范

推荐保留的顶部说明：
- 脚本用途
- 参考脚本
- 与参考脚本的差异
- 可通过环境变量覆盖的核心控制项
- `EXIT_INTERVAL` / `ENABLE_TEST_TRAIN_RUN` 等常用调试方式

新增或派生训练脚本时，顶部说明默认参考同家族最近的 launcher，并按下面顺序补齐信息：
- 脚本用途
- `Reference launcher`
- `Difference from the reference`
- 可通过环境变量覆盖的分布式设置
- 常用调试方式或 `Optional` 覆盖项
- 若脚本是 smoke / 10-step / 验证脚本，再补充 `Default verification config`

如果默认值为了显存、安全性或资源约束而特意收紧，必须在顶部注释中明确写出：
- 为什么收紧
- 收紧后的关键默认值
- 该默认值是面向哪类机器或验证场景

变量默认值规范：
- 用户可覆盖的控制变量，优先使用 `${VAR:-default}`。
- 固定实验脚本中的硬编码值可以直接写死，但应保证同一类脚本风格一致。

注释规范：
- 解释某段代码用途时，使用普通注释即可。
- 只有 5 个主分类使用 `# ========== ... ==========`。
- 不要在同一个脚本里混用新旧分隔线命名。
- 派生脚本应先复用参考脚本的注释骨架，再只改动与当前实验真实差异相关的条目。
- 注释必须与脚本当前默认值保持一致，不能复制参考脚本后遗留过期描述。

## 五、例外情况

以下情况允许局部不完全对称，但应尽量保持可读性：
- checkpoint 路径选择依赖 `TP_SIZE` 或其他模型参数  
  例如先定义 `TP_SIZE`，再根据 `TP_SIZE` 选择 `LOAD_CHECKPOINT_PATH`。
- tee 日志初始化依赖 `LOG_DIR`  
  可以把实际 tee 代码块放在路径段后半部分。
- 个别历史脚本为了兼容老实验命名，保留少量旧变量名  
  但新增脚本不要再引入新的命名分叉。

## 六、自检要求

整理或新增脚本后，至少做以下检查：
- `bash -n <script>`
- 确认旧分隔线已清理干净
- 确认训练段里：
  `LR` / `MIN_LR` 在派生变量之前
  派生变量之间的计算关系不变
- 确认最终启动命令参数未被误改

## 七、当前目录脚本家族

- `qwen3_0p6b_baseline_*.sh`：baseline 对照训练
- `qwen3_0p6b_rmt_*.sh`：RMT 训练
- `qwen3_0p6b_armt_*.sh`：ARMT 训练
- `qwen3_0p6b_armt_ct_from_mlm_ckpt_*.sh`：从 MLM checkpoint 派生的 ARMT 验证脚本

后续新增脚本时，默认直接复用这份规范，不再重新发明分段方式。
