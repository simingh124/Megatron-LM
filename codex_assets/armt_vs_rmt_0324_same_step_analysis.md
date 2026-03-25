# ARMT vs RMT 0324 实验同 Step 对比分析

## 1. 结论摘要

本报告只比较同 step 下的结果，不考虑同 wall-clock。

在本次分析快照下：

- `output_logs` 共同 step 上界是 `14936`
- `tensorboard` 共同 step 上界是 `14902`
  - 原因是 ARMT 的 event 文件比日志慢了 `34` step 刷新

结论很明确：

1. 在相同训练步数下，RMT 的 `lm loss` 一直略优于 ARMT，不是只在后期才反超。
2. 这种优势在 `output_logs`、tensorboard 主标量、`train/chunk_00_loss`、`train/chunk_01_loss` 上是一致的。
3. 从细粒度指标看，ARMT 当前配置下更像是“memory 机制没有充分发挥作用”，而不是训练直接崩掉：
   - 没有出现 NaN，`loss-scale` 始终为 `1.0`
   - 但 `retrieved_to_hidden_ratio` 很低，说明检索出来的记忆对主干 hidden state 的影响偏弱
   - `mem_token_cosine_mean` 偏高，说明 memory token 有明显趋同/塌缩迹象
4. RMT 的 memory token 状态更像“已经在工作”：
   - read/write token norm 相对 context 更强
   - read/write token 之间的平均 cosine 接近 `0`，比 ARMT 更分散
   - 两个 chunk 的 loss 都比 ARMT 更低

因此，本次结果更像是：

- “在当前 0324 这组配置下，RMT 同 step 更好”

而不是：

- “已经证明 ARMT 机制天然不如 RMT”

## 2. 数据来源与分析口径

### 2.1 相关脚本

- `playground/rmt/qwen3_0p6b_armt_0324_fs_wo_tbptt.sh`
- `playground/rmt/qwen3_0p6b_rmt_0324_fs_wo_tbptt.sh`

### 2.2 日志与 tensorboard 路径

- `output_logs`
  - `/mnt/step3-abla/siming/exp_logs/output_logs/rmt_qwen/qwen3_0p6b_armt_0324_fs_wo_tbptt`
  - `/mnt/step3-abla/siming/exp_logs/output_logs/rmt_qwen/qwen3_0p6b_rmt_0324_fs_wo_tbptt`
- `tensorboard`
  - `/mnt/step3-abla/siming/exp_logs/tensorboard/rmt_qwen/qwen3_0p6b_armt_0324_fs_wo_tbptt`
  - `/mnt/step3-abla/siming/exp_logs/tensorboard/rmt_qwen/qwen3_0p6b_rmt_0324_fs_wo_tbptt`
- `tb_to_json.py` 导出结果
  - `/mnt/step3-abla/siming/exp_logs/tb_infos/rmt_qwen/qwen3_0p6b_armt_0324_fs_wo_tbptt.json`
  - `/mnt/step3-abla/siming/exp_logs/tb_infos/rmt_qwen/qwen3_0p6b_rmt_0324_fs_wo_tbptt.json`

### 2.3 快照时实验状态

分析时两个实验都仍在运行。

| 实验 | 最新 log step | 说明 |
| --- | ---: | --- |
| ARMT | 14936 | 本次 `output_logs` 同 step 上界 |
| RMT | 30768 | 比 ARMT 跑得更远，但本报告不使用这一点做优劣结论 |

### 2.4 本报告使用的比较区间

| 数据源 | 同 step 上界 | 说明 |
| --- | ---: | --- |
| `output_logs` | 14936 | 按两边日志最新共同 step 对齐 |
| `tensorboard` | 14902 | ARMT event flush 比日志慢 34 step，因此 TB 主指标和细粒度指标统一截到 14902 |

## 3. 脚本与配置差异

这次对比不是完全等价的 apples-to-apples。

| 项目 | ARMT | RMT | 影响 |
| --- | --- | --- | --- |
| 训练入口 | `examples/armt/train.py` | `examples/rmt/train.py` | 实现路径不同 |
| memory 机制 | layer-level associative memory | model-level recurrent memory tokens | 机制本身不同 |
| `MICRO_BATCH_SIZE` | `20` | `30` | 次要公平性差异；在 `GLOBAL_BATCH_SIZE` 一致时通常不是主因 |
| `GLOBAL_BATCH_SIZE` | `480` | `480` | 对齐 |
| `NUM_MEM_TOKENS` | `16` | `16` | 对齐 |
| chunk size | `512` | `512` | 对齐 |
| `--no-recurrent-tbptt-mode` | 开启 | 开启 | 对齐 |
| ARMT 专属 | `--armt-n-heads 16` | 无 | ARMT 额外复杂度更高 |
| rank0 参数量 | `684,343,744` | `595,804,160` | ARMT 多 `88,539,584` 参数，约 `1.149x` |

参数量差异是这次分析里必须明确点出来的事实。当前对比并不是“参数量几乎一致，仅替换 memory 机制”。

## 4. output_logs 同 Step 对比

### 4.1 EMA 对比

使用工具：

- `/mnt/step3-abla/siming/code_repo/train_tools/exp_calc/log_loss_ema.py`
- `beta=0.9`
- `end-step=14936`

结果如下：

| 实验 | step 区间 | `lm loss` EMA | `grad norm` EMA | `learning rate` EMA |
| --- | --- | ---: | ---: | ---: |
| ARMT | `[1, 14936]` | 2.79944 | 0.134693 | 0.000500 |
| RMT | `[1, 14936]` | 2.79319 | 0.138483 | 0.000500 |

同 step 下，RMT 的 `lm loss EMA` 低 `0.00625`。

### 4.2 同 step 区间平均值

| 实验 | `lm loss` mean | `grad norm` mean |
| --- | ---: | ---: |
| ARMT | 3.17155 | 0.22950 |
| RMT | 3.11575 | 0.32139 |

这里最重要的是 `lm loss` 平均值：RMT 低 `0.05580`。  
也就是说不只是最后几百步 EMA 略好，整个 `[1, 14936]` 区间上，RMT 的平均 loss 也更低。

### 4.3 关键 step 原始值

这些 step 下两边 `consumed samples` 完全一致，因此是严格同训练进度对比。

| Step | ARMT `lm loss` | RMT `lm loss` | ARMT - RMT | ARMT `grad norm` | RMT `grad norm` |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1000 | 3.673203 | 3.657068 | 0.016135 | 0.288 | 0.274 |
| 2000 | 3.351547 | 3.281307 | 0.070240 | 0.151 | 0.157 |
| 4000 | 3.087478 | 3.075103 | 0.012375 | 0.131 | 0.148 |
| 8000 | 2.926800 | 2.923282 | 0.003518 | 0.146 | 0.154 |
| 12000 | 2.814557 | 2.814354 | 0.000203 | 0.131 | 0.211 |
| 14000 | 2.787204 | 2.785002 | 0.002202 | 0.129 | 0.126 |
| 14936 | 2.770266 | 2.762521 | 0.007745 | 0.130 | 0.131 |

观察：

- 在这 7 个采样点上，RMT 的 `lm loss` 全部低于 ARMT。
- `12000` step 两者非常接近，但仍是 RMT 更低。
- 这说明 RMT 的领先不是某个局部抖动，而是从早期就存在并持续保持。

## 5. TensorBoard 主指标同 Step 对比

### 5.1 tag 覆盖情况

| 项目 | 数量 |
| --- | ---: |
| ARMT TB tags | 27 |
| RMT TB tags | 34 |
| 公共 tags | 14 |
| ARMT-only tags | 13 |
| RMT-only tags | 20 |

两边公共主标量主要包括：

- `lm loss`
- `grad-norm`
- `learning-rate`
- `loss-scale`
- `train/chunk_00_loss`
- `train/chunk_01_loss`

### 5.2 主标量统计

这里统一截到 `tb_common_end = 14902`。

| 指标 | ARMT mean | RMT mean | ARMT EMA | RMT EMA | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| `lm loss` | 3.17239 | 3.11648 | 2.80121 | 2.79569 | RMT 更低 |
| `grad-norm` | 0.22972 | 0.32181 | 0.13507 | 0.13778 | 数值接近，RMT 略高 |
| `learning-rate` | 0.00048324 | 0.00048324 | 0.00050000 | 0.00050000 | 完全一致 |
| `loss-scale` | 1.00000 | 1.00000 | 1.00000 | 1.00000 | 完全一致 |

`loss-scale` 始终是 `1.0`，说明两边都没有明显数值稳定性问题。  
因此这次 ARMT 不占优，不是因为训练直接发散或出现 NaN。

### 5.3 分 chunk loss 对比

这两个 tag 很关键，因为本实验 `seq_length=1024`、chunk size=`512`，所以可以看两个 chunk 的 loss 状态。

| 指标 | ARMT mean | RMT mean | ARMT EMA | RMT EMA | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| `train/chunk_00_loss` | 3.20726 | 3.15033 | 2.84344 | 2.84017 | RMT 更低 |
| `train/chunk_01_loss` | 3.13752 | 3.08263 | 2.75897 | 2.75121 | RMT 更低 |

这里的信号很重要：

- RMT 不只是总 `lm loss` 更低
- 第一段 chunk 和第二段 chunk 都更低
- 第二段 chunk 本应更依赖 memory 机制带来的跨 chunk 信息，RMT 依然更好

这直接削弱了“ARMT 机制应该在跨 chunk 记忆上明显更占优”的预期。

## 6. TensorBoard 细粒度训练状态

### 6.1 ARMT 专属指标

统一截到 `tb_common_end = 14902`。

| ARMT 指标 | mean | EMA | last | 解释 |
| --- | ---: | ---: | ---: | --- |
| `armt/read/retrieved_to_hidden_ratio` | 0.02725 | 0.02653 | 0.02579 | 检索到的 memory 只占 hidden norm 的约 2.7% |
| `armt/token/mem_ctx_norm_ratio` | 0.66558 | 0.66071 | 0.66060 | memory token norm 只有 context token 的约 66% |
| `armt/token/mem_token_cosine_mean` | 0.44672 | 0.57260 | 0.57451 | memory token 平均相似度偏高 |
| `armt/token/mem_token_cosine_gt_0p8_ratio_mean` | 0.14920 | 0.28329 | 0.28483 | 高相似 token 对占比偏高 |
| `armt/write/write_gate_mean` | 0.24940 | 0.26432 | 0.26452 | 写门整体偏保守 |

这些指标组合起来的含义比较一致：

1. `retrieved_to_hidden_ratio` 很低  
   说明 ARMT 虽然引入了 associative memory，但在当前训练阶段，真正加回主干 hidden state 的检索信息很弱。

2. `mem_ctx_norm_ratio < 1`  
   说明 memory token 的幅值整体弱于 context token，本身就不像一个很强的信号通道。

3. `mem_token_cosine_mean` 和 `gt_0.8_ratio` 都偏高  
   说明不同 memory token 之间有明显趋同，存在一定“token collapse / 冗余化”迹象。  
   换句话说，ARMT 额外给了 16 个 memory token，但这些 token 并没有体现出很强的多样性。

综合判断：  
ARMT 当前更像是“memory 分支在训练，但作用偏弱，而且 memory token 内部相似度偏高”，这会直接削弱它相对 RMT 的理论优势。

### 6.2 RMT 专属指标

同样截到 `tb_common_end = 14902`。

| RMT 指标 | mean | EMA | last | 解释 |
| --- | ---: | ---: | ---: | --- |
| `rmt/token/read_ctx_norm_ratio` | 1.23044 | 1.32375 | 1.32783 | read memory norm 明显强于 context |
| `rmt/token/write_ctx_norm_ratio` | 1.20909 | 1.26041 | 1.26476 | write memory norm 也强于 context |
| `rmt/token/read_write_norm_ratio` | 1.02962 | 1.08415 | 1.08398 | read/write 两侧幅值接近，整体比较平衡 |
| `rmt/token/read_mem_token_cosine_mean` | -0.00982 | -0.00055 | -0.00072 | read memory token 平均相似度接近 0 |
| `rmt/token/write_mem_token_cosine_mean` | 0.01096 | 0.05933 | 0.05576 | write memory token 平均相似度也较低 |
| `rmt/token/read_mem_token_cosine_gt_0p8_ratio_mean` | 0.18881 | 0.23718 | 0.23923 | 仍有部分高度相似对，但整体没塌到同一方向 |
| `rmt/token/write_mem_token_cosine_gt_0p8_ratio_mean` | 0.17029 | 0.23411 | 0.23848 | write memory 也是类似趋势 |

这里最关键的两个判断是：

1. RMT 的 read/write memory token 是“有力度的”  
   `read_ctx_norm_ratio`、`write_ctx_norm_ratio` 都在 `1.2+`，说明 memory token 不是很弱的小分支，而是确实在和 context 交互。

2. RMT 的 memory token 比 ARMT 更分散  
   read/write 的平均 cosine 都接近 `0`，而 ARMT 的 memory token cosine mean 已经到 `0.57` 左右。  
   这说明 RMT 的 16 个 memory token 在表示空间里更分散，不容易退化成一组高度相似的 token。

综合判断：  
RMT 当前的 memory channel 更“活跃”，也更“有区分度”。

## 7. 为什么这次同 Step 下是 RMT 更好

### 7.1 现象层面

现象本身没有歧义：

- `output_logs` 同 step EMA：RMT 更低
- `output_logs` 关键 step 原始值：RMT 在所有采样点都更低
- tensorboard `lm loss`、`chunk_00_loss`、`chunk_01_loss`：RMT 更低

所以本次 0324 结果不能解释成“只是 RMT 跑得更久，所以看起来更好”。

### 7.2 更可能的原因

#### 原因 1：当前对比不是严格等价设置

最明显的两点：

- `MICRO_BATCH_SIZE` 不同：ARMT `20`，RMT `30`
- 参数量不同：ARMT 比 RMT 多 `88.54M` 参数

这意味着：

- `MICRO_BATCH_SIZE` 的差异会让实验不属于“完全逐项相同”的严格对照
- 但在当前设置下，`GLOBAL_BATCH_SIZE` 一致、dropout 为 0、也没有 BatchNorm 一类依赖 microbatch 统计的层，因此 `MICRO_BATCH_SIZE` 更可能只是次要噪声，而不是当前 ARMT 落后的主因
- 更关键的非等价项其实是 ARMT 还要在同样训练步数下，去优化更大的一组参数

因此这次结果并不能单独归因为“ARMT 机制不如 RMT”；但若要追主因，优先级也应放在参数量、首 chunk 读路径和机制利用率上，而不是 `MICRO_BATCH_SIZE`。

#### 原因 2：ARMT 的 memory 分支当前利用率偏低

ARMT 最关键的负面信号是：

- `retrieved_to_hidden_ratio` 只有 `~0.0265`
- `mem_ctx_norm_ratio` 只有 `~0.6607`

这意味着：

- 检索出来的 memory 信息进入主干时很弱
- memory token 本身也比 context token 更弱

换句话说，ARMT 的额外 memory 结构虽然存在，但到 `14902` step 时，它对主干计算的实际“话语权”还不够高。

#### 原因 3：ARMT memory token 有明显趋同

ARMT：

- `mem_token_cosine_mean EMA = 0.5726`
- `mem_token_cosine_gt_0p8_ratio_mean EMA = 0.2833`

RMT：

- `read_mem_token_cosine_mean EMA = -0.0005`
- `write_mem_token_cosine_mean EMA = 0.0593`

这组对比说明：

- ARMT 的 memory token 更容易往相似方向收缩
- RMT 的 memory token 更分散，内部表示更有区分度

如果 memory token 之间越来越像，它们能提供的额外信息容量就会下降，理论优势也就很难落到 loss 上。

#### 原因 4：ARMT 当前配置更难从 scratch 优化

当前脚本里 ARMT 还额外指定了：

- `--armt-n-heads 16`

同时 ARMT 的实现是 layer-level associative memory，会在各层引入更复杂的 retrieval / update 路径。  
相比之下，RMT 是 model-level recurrent memory token，路径更直接。

从这次训练状态看，更合理的推断是：

- ARMT 这套配置从零训练时优化难度更高
- 到当前训练进度，额外结构的收益还没有抵消它带来的优化负担

### 7.3 结合论文和当前代码，为什么这版 ARMT 更容易输给 RMT

下面这部分是更深一层的分析。结论先说：

- 我没有看到一个“明显实现 bug”可以单独解释 ARMT 输掉
- 更像是当前实现方式、训练任务和训练配方，整体上更偏向 RMT 的优势区间

#### 原因 5：论文验证的优势区间，本来就更接近“超长 sparse retrieval”，而不是当前这类从零开始的 web-LM 预训练

ARMT 论文摘要写得很明确，主结论是：

- ARMT 的设计目标是“very long sequences + constant-time processing”
- 主要结果是“associative retrieval tasks”和“BABILong over 50M tokens”

参考：

- ARMT 论文摘要：<https://arxiv.org/abs/2407.04841>

而 RMT 论文摘要则明确把 language modeling 写进了主结果：

- RMT 在 language modeling 上与 Transformer-XL 持平
- 在更长序列处理任务上优于 Transformer-XL

参考：

- RMT 论文摘要：<https://arxiv.org/abs/2207.06881>

这两篇论文的“强证据区间”并不一样：

- RMT 的主论文已经明确覆盖 LM
- ARMT 的主论文强项是 associative retrieval / BABILong / 超长上下文单事实查找

所以当前这个实验：

- Qwen3-0.6B
- FineWeb 风格自然文本
- 从零初始化
- 目标是 next-token language modeling

本身就更靠近 RMT 已经被验证过的 regime，而不是 ARMT 论文最强的 regime。

这并不意味着 ARMT 不能做 LM，而是说明：

- “论文里 ARMT 比 RMT 更强”主要成立在长距离检索任务上
- 不能直接平移成“从零开始的大规模自然语言预训练，ARMT 一定比 RMT 更强”

这个差异其实也能从官方代码仓库看出来。ARMT 官方仓库 README 里最醒目的表述仍然是：

- 50M token extrapolation
- BABILong SOTA

同时它给的语言建模示例是：

- `finetune_armt_llama3.2_pg19_sliding.sh`

参考：

- ARMT 官方仓库 README：<https://github.com/RodkinIvan/associative-recurrent-memory-transformer>

也就是说，作者公开给出的 LM 示例更接近“sliding-window finetuning”，而不是我们现在这种“从零开始预训练 Qwen3-0.6B”。

#### 原因 6：当前训练样本只有 2 个 recurrent chunks，ARMT 的优势几乎没有展开空间

当前脚本里：

- `SEQ_LENGTH=1024`
- `ARMT_CHUNK_SIZE=512`
- `RECURRENT_CHUNK_SIZE=512`

所以每个样本只有 2 个 chunks。

这件事对 RMT 和 ARMT 的影响不一样：

1. RMT 只需要做一件相对简单的事  
   把上一 chunk 的 read/write memory tokens 传给下一 chunk。

2. ARMT 的真正优势，要在“很多 chunks、很多 recurrent hops、很长距离 sparse retrieval”时才更容易体现出来  
   但现在每个样本只有一次跨 chunk handoff，它的 associative memory 几乎没有施展空间。

换句话说：

- RMT 在 2-chunk setting 下，已经够用
- ARMT 在 2-chunk setting 下，反而更像是“为远距离检索设计的重型机制，被拿来做只跨 1 次边界的信息传递”

这和 ARMT 官方仓库 README 里强调的“trained only on 16k, scale up to 50M tokens”是相反的使用区间。论文的强项是 many-hop long-range retrieval，而不是 2-hop。

#### 原因 7：当前实现里，ARMT 的第一个 chunk 天然没有 memory read；RMT 则有

这是当前代码里最关键、最具体的结构差异。

ARMT：

- `ARMTModel._concat_memory_embeddings()` 只把 memory embeddings 追加到序列尾部
- `ARMTModel._adjust_attention_mask()` 用的是标准因果 mask
- 因为 memory tokens 在 context 后面，context token 不能看未来位置，所以它们不能在同一个 chunk 内被 context 读取
- ARMT 真正的“读 memory”路径只能来自 `AssociativeLayer.associate()`
- 但 `associate()` 在 `_first_chunk` 时直接返回全零

对应代码：

- `megatron/core/models/armt/armt_model.py`
  - `_concat_memory_embeddings()`：把 `mem` 接在 `decoder_input` 后面
  - `_adjust_attention_mask()`：对 `seq_len + num_mem_tokens` 做标准 causal mask
- `megatron/core/models/armt/associative_layer.py`
  - `if self._first_chunk: result = torch.zeros_like(hidden_states)`

RMT：

- `RMTModel._get_memory_state()` 在 `memory_state is None` 时，会回退到 learned `memory_embeddings`
- `RMTModel._concat_memory_embeddings()` 默认把 read memory tokens 放在序列前面、write memory tokens 放在序列后面
- 所以第一 chunk 里，context token 一开始就能读到 read memory tokens

对应代码：

- `megatron/core/models/rmt/rmt_model.py`
  - `_get_memory_state()`
  - `_concat_memory_embeddings()`

这意味着在当前实现中：

- RMT 的 chunk 0 有 read path
- ARMT 的 chunk 0 没有 recurrent read path

而我们的训练调度默认又没有开启 `--no-loss-from-first-chunk`，所以 chunk 0 的 loss 也被完整计入训练目标。

对应代码：

- `megatron/core/pipeline_parallel/recurrent_schedules.py`
  - 只有显式传 `--no-loss-from-first-chunk` 才会把首 chunk 的 `loss_mask` 清零
  - 当前脚本没有传这个参数

这和 tensorboard 结果是对得上的：

- `train/chunk_00_loss`：RMT 更低
- `train/chunk_01_loss`：RMT 还是更低

也就是说，ARMT 不只是第二块没追回来，第一块在实现上就先天更吃亏。

#### 原因 8：ARMT 的读路径比 RMT 更“窄”、更弱，也更难优化

ARMT 当前实现的读路径是：

1. 对当前 hidden 做 `W_mq`
2. 过 DPFP `phi`
3. 与 `W_mem / z` 做 associative retrieval
4. 把检索结果作为一个 additive residual，加回当前 hidden
5. 然后再进入正常的 attention + MLP

对应代码：

- `megatron/core/models/armt/armt_layer.py`
  - `retrieved = self.associative_layer.associate(...)`
  - `hidden_states = hidden_states + retrieved`

RMT 的读路径则更直接：

- read memory tokens 直接作为序列前缀参与 self-attention
- context token 通过标准 attention 机制读取 memory token
- 这条路径和普通 Transformer 的 inductive bias 是一致的

这两种路径的差别是：

1. RMT 的记忆读取发生在标准 self-attention 里面  
   模型比较容易沿用已有的 Transformer 表达能力。

2. ARMT 的记忆读取是 attention 之前的一条额外旁路  
   它依赖 `W_mq/W_mk/W_mv`、DPFP 映射、归一化、校正项、write gate 一起配合工作，学习难度显著更高。

这和监控指标一致：

- `armt/read/retrieved_to_hidden_ratio ≈ 0.0265`
- 说明 ARMT 检索出来的内容，相对主干 hidden 非常弱

相比之下，RMT：

- `read_ctx_norm_ratio ≈ 1.32`
- `write_ctx_norm_ratio ≈ 1.26`

说明 RMT 的 memory token 真正在参与主干表示，而不是很弱的小残差支路。

#### 原因 9：ARMT 当前实现是“28 个 layer-local recurrent memories”，而 RMT 只有“1 个 model-level recurrent memory”

这一点很容易被忽略，但实际上非常重要。

RMT：

- 只有一个 `memory_state`
- 每个样本跨 chunk 时，整个模型只需要维护一套 recurrent state

ARMT：

- `reset_all_memory()` 会遍历所有 `ARMTLayer`
- 每一层都有自己的 associative memory 状态 `W_mem` / `z`
- 当前 28 层模型，相当于有 28 套 layer-local recurrent states 同时学习

在“每个样本只有 2 chunks”的前提下，这意味着：

- RMT 只要学会 1 套跨 chunk 读写规则
- ARMT 要让 28 层各自都学会一套有用的读写规则

这对优化非常不友好，尤其还是从零训练。

#### 原因 10：当前 ARMT 配置把 associative memory 做得很重，但训练配方又是最难的 from-scratch

当前实现里：

- `--armt-d-mem` 默认是 `hidden_size`
- 当前实验 `hidden_size=1024`

这会让每层都新增至少这些投影：

- `W_mq: 1024 x 1024`
- `W_mk: 1024 x 1024`
- `W_mv: 1024 x 1024`
- `W_mb: 1024 x 16`

按当前配置估算，每层额外参数约：

- `3,162,128`

28 层总计就是：

- `88,539,584`

这与日志里观测到的参数差值完全一致：

- ARMT rank0 参数量：`684,343,744`
- RMT rank0 参数量：`595,804,160`
- 差值：`88,539,584`

也就是说，当前 ARMT 不是“很轻地加一层 memory 机制”，而是在每层都加了一套相当重的 associative projections。

如果这是在一个已经训练好的强 baseline 上做增量微调，问题可能没那么大。  
但当前脚本明确是：

- 不加载 checkpoint
- backbone 和 ARMT/RMT 参数都从零初始化

对应脚本注释：

- `playground/rmt/qwen3_0p6b_armt_0324_fs_wo_tbptt.sh`
- `playground/rmt/qwen3_0p6b_rmt_0324_fs_wo_tbptt.sh`

因此当前训练其实是在做一件很难的事：

- 让一个更重、更复杂、读写路径更间接的 ARMT，从头学会语言模型主干和 associative memory 两件事

而 RMT 要学的东西更少、更接近标准 Transformer。

#### 原因 11：当前细粒度指标说明 ARMT 的 memory token 已经开始冗余化

如果 ARMT 真正在发挥论文里的 associative retrieval 优势，通常应当看到：

- memory token 有较好的分工
- retrieval 对 hidden 有足够强的影响
- 第二个 chunk 能明显受益

但目前看到的是：

- `retrieved_to_hidden_ratio` 很低
- `mem_ctx_norm_ratio < 1`
- `mem_token_cosine_mean` 高达 `0.57` 左右
- `mem_token_cosine_gt_0p8_ratio_mean` 到了 `0.28` 左右

这更像是：

- memory token 有一部分在往相似方向收缩
- associative read/write 通路没有形成强而分化的表征

而 RMT 的 read/write memory token：

- 平均 cosine 更接近 `0`
- read/write 与 context 的 norm ratio 更稳定地大于 `1`

所以从训练状态上看，RMT 更像已经学会“怎么用 memory”；ARMT 则更像“memory 结构在，但还没有学会高效使用它”。

### 7.4 更合理的总体判断

基于论文、官方仓库和当前代码，我认为更合理的判断是：

1. 当前结果并不表明论文错了。  
   论文证明的是 ARMT 在 associative retrieval / BABILong / 超长 sparse recall 场景里的优势。

2. 当前结果更像是“这份 Megatron 实现 + 这组 0324 配置 + 这个训练任务”不适合让 ARMT 发挥论文优势。  
   尤其是：
   - 只有 2 chunks
   - 从零训练
   - 第一个 chunk 没有 ARMT recurrent read
   - `--no-loss-from-first-chunk` 没开
   - associative branch 很重，但利用率偏低
   - `MICRO_BATCH_SIZE` 不一致属于次要公平性差异，但不是这次结果的核心解释

3. 在这个设定里，RMT 的 inductive bias 更直接、更接近标准 Transformer，也更接近已被论文验证的 LM regime，所以它现在赢是合理的。

## 8. 结论

在本次快照下，只看同 step：

1. RMT 确实比 ARMT 更好。
2. 这种优势在主 loss、分 chunk loss 和细粒度 memory 状态上都是一致的。
3. ARMT 当前主要问题不像“训练崩了”，而更像“memory 分支作用偏弱，且 memory token 多样性不足”。
4. 本次结果更支持下面这个判断：

> 在当前 0324 这组超参与实现配置下，ARMT 没有发挥出理论优势，RMT 的有效训练状态更好。

而不支持下面这个更强的判断：

> ARMT 机制本身一定不如 RMT。

再进一步说，当前结果更像是下面这个组合问题：

- 任务与论文主验证区间不一致
- 当前实现里 ARMT 的第一 chunk 没有 recurrent read
- 当前训练只有 2 个 chunks，recurrent depth 太浅
- 当前 ARMT 从零训练且额外参数太重

这四点叠加，足以让 ARMT 在这个实验里输给 RMT，而不需要假设论文结论失效。

## 9. 下一步建议

如果要继续定位原因，我建议优先做下面几件事：

1. 把 `MICRO_BATCH_SIZE` 对齐，作为公平性收尾项  
   这更适合用来消除次要噪声，而不是验证当前主因。当前结果的主要解释，不在这里。

2. 单独扫 `--armt-n-heads`  
   至少测试 `1 / 2 / 4 / 8 / 16`，观察：
   - `retrieved_to_hidden_ratio`
   - `mem_token_cosine_mean`
   - `mem_token_cosine_gt_0p8_ratio_mean`
   - `train/chunk_01_loss`

3. 优先扫 `--armt-d-mem`，不要默认直接等于 `hidden_size`  
   这是当前 ARMT 额外参数暴涨的核心来源。建议至少测试：
   - `64 / 128 / 256 / 512 / 1024`
   - 重点看 `retrieved_to_hidden_ratio` 是否上升，以及 `mem_token_cosine_mean` 是否下降

4. 对 ARMT 单独试一版 `--no-loss-from-first-chunk`  
   当前实现里第一 chunk 没有 associative recurrent read，把首 chunk loss 继续算进去，会天然压低 ARMT 的训练信号质量。

5. 把 recurrent depth 做深再比  
   当前只有 2 chunks，太不利于 ARMT。建议至少做一版：
   - `seq_length=2048, chunk_size=512`，得到 4 chunks
   - 或更长序列，创造更多跨 chunk hops

6. 不要只做 from-scratch，再做一版“先 baseline 后切 ARMT / RMT”的实验  
   当前官方 ARMT 仓库给出的 LM 示例更接近 sliding-window finetuning。  
   如果论文优势需要先有一个成熟语言模型主干，再学习 memory 机制，那么直接从零训练会显著吃亏。

7. 控制参数量再比一次  
   当前 ARMT 多约 `14.9%` 参数，最好做一版更接近参数量的公平对照。

8. 继续盯细粒度状态而不是只看 `lm loss`  
   当前最有解释力的指标已经很明确：
   - ARMT：`retrieved_to_hidden_ratio`、`mem_ctx_norm_ratio`、`mem_token_cosine_mean`
   - RMT：`read_ctx_norm_ratio`、`write_ctx_norm_ratio`、`read/write_mem_token_cosine_mean`

## 10. 参考资料

- ARMT 论文摘要：<https://arxiv.org/abs/2407.04841>
- RMT 论文摘要：<https://arxiv.org/abs/2207.06881>
- ARMT 官方仓库 README：<https://github.com/RodkinIvan/associative-recurrent-memory-transformer>
- RMT 官方仓库 README：<https://github.com/booydar/LM-RMT>
