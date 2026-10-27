# Zero Bubble Pipeline Parallelism

本文介绍 Megatron-LM BR adaptor 当前 Zero Bubble（ZB）实现的计算拆分方式、调度和通信流程、配置参数及使用限制。该功能通过 `--enable-zero-bubble` 开启，目前属于实验性功能。

## 作用和适用场景

Zero Bubble 的目标是减少 pipeline parallelism（PP）中等待上下游数据产生的计算空档。普通非交错 1F1B 把一次 backward 作为整体执行；ZB 将其中可延后的 weight-gradient 计算单独调度，让影响上游推进的 input-gradient 计算先执行，再用延后的工作填补其他位置的空档。

它适合已经使用 PP、希望研究流水线气泡和计算调度的训练场景。实际收益受 microbatch 数量、各 stage 的计算负载、通信和同步耗时，以及延迟计算所需的显存共同影响。

当前实现提供 F/B/W 拆分、排程和运行时执行，迭代结束后仍沿用 Megatron 的梯度收尾和 optimizer step。论文中的 optimizer post-validation / rollback 优化没有在这里接入。因此，开启该功能后，仍需分别测量计算调度空档、通信等待和迭代边界开销。

## F/B/W 拆分原理

### F、B 和 W 分别表示什么

以线性层 `Y = X A^T` 为例：

| 标记 | 工作 | 主要依赖和用途 |
| --- | --- | --- |
| F | Forward，计算输出 `Y` | 依赖当前 microbatch 的输入和模型参数；输出发送到下一个 PP stage。 |
| B | Input-gradient backward，计算 `dX = dY A` | 依赖下游传回的输出梯度；输入梯度需要继续传给上一个 PP stage。 |
| W | Weight-gradient，计算并累积 `dA = dY^T X` | 依赖保存的输入和输出梯度；必须在本轮梯度收尾及参数更新前完成。 |

普通 1F1B 中，B 和 W 在同一次 backward 调用中完成，之后 runtime 才提交该 microbatch 的输入梯度发送。ZB 则允许先完成 B，并将 W 保存为待执行任务，在后续调度到 W 节点时执行。

上述拆分的依据是：上游继续 backward 需要 `dX`，而当前 stage 的 `dA` 可以稍后计算。这也是 [Zero Bubble 论文](https://arxiv.org/abs/2401.10241)的核心调度思路。

本仓库的 W 主要对应适配后的 local Linear weight-gradient 计算。其他 autograd backward 工作仍在 B 阶段执行；例如 bias gradient 没有通过这条路径延后。不能将 W 理解为全模型所有参数梯度统一移动到了独立阶段。

### 延迟 W 与显存的关系

延迟 W 需要保留计算它所需的输入、输出梯度及相关引用，直到对应 W 任务执行结束。因此，减少调度空档可能伴随更长的张量存活时间，需关注实际峰值显存。

当前 scheduler 的 `mem_f`、`mem_b`、`mem_w` 是抽象代价，不能直接解释为实际释放或分配了多少设备内存。特别是 B 完成后，延迟 W 所需的张量仍可能存活。

### PP=4、8 个 microbatch 的编排对比

下面采用 `TP=1`、`PP=4`、`DP=1`，例如 `GBS=32`、`MBS=4`，每轮有 8 个 microbatch，编号为 `0`～`7`。GPU0 是第一个 stage，GPU3 是最后一个 stage。

两张表均从左向右阅读，每列表示相同的相对计算单位，假设 F、B、W 等代价，并忽略通信和同步开销。同一列表示可以并行执行的计算；`·` 表示计算空档，包括流水线启动、依赖等待和收尾。表中只展示计算编排，不标实际耗时。

- `Fi`：第 `i` 个 microbatch 的 forward。
- `Bi`：第 `i` 个 microbatch 的 input-gradient backward。
- `Wi`：第 `i` 个 microbatch 的 weight-gradient。

#### 普通非交错 1F1B

普通 1F1B 的 `Bi`、`Wi` 在同一次 backward 调用中连续完成，之后才向上游传递输入梯度。这里将它们展开为相邻两格，便于和 ZB 对照。

| 设备 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| GPU0 | `F0` | `F1` | `F2` | `F3` | `·` | `·` | `·` | `·` | `·` | `·` | `B0` | `W0` | `F4` | `B1` | `W1` | `F5` | `B2` | `W2` | `F6` | `B3` | `W3` | `F7` | `B4` | `W4` | `·` | `B5` | `W5` | `·` | `B6` | `W6` | `·` | `B7` | `W7` |
| GPU1 | `·` | `F0` | `F1` | `F2` | `·` | `·` | `·` | `·` | `B0` | `W0` | `F3` | `B1` | `W1` | `F4` | `B2` | `W2` | `F5` | `B3` | `W3` | `F6` | `B4` | `W4` | `F7` | `B5` | `W5` | `·` | `B6` | `W6` | `·` | `B7` | `W7` | `·` | `·` |
| GPU2 | `·` | `·` | `F0` | `F1` | `·` | `·` | `B0` | `W0` | `F2` | `B1` | `W1` | `F3` | `B2` | `W2` | `F4` | `B3` | `W3` | `F5` | `B4` | `W4` | `F6` | `B5` | `W5` | `F7` | `B6` | `W6` | `·` | `B7` | `W7` | `·` | `·` | `·` | `·` |
| GPU3 | `·` | `·` | `·` | `F0` | `B0` | `W0` | `F1` | `B1` | `W1` | `F2` | `B2` | `W2` | `F3` | `B3` | `W3` | `F4` | `B4` | `W4` | `F5` | `B5` | `W5` | `F6` | `B6` | `W6` | `F7` | `B7` | `W7` | `·` | `·` | `·` | `·` | `·` | `·` |

#### Zero Bubble

ZB 先执行影响上游推进的 B，再安排 W。以下编排中，GPU0 用 W 填入后续 B 之间的位置，GPU1、GPU2 逐步延后 W，GPU3 则先连续执行 F/B，再集中完成 W。

| 设备 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| GPU0 | `F0` | `F1` | `F2` | `F3` | `F4` | `F5` | `F6` | `B0` | `F7` | `W0` | `B1` | `W1` | `B2` | `W2` | `B3` | `W3` | `B4` | `W4` | `B5` | `W5` | `B6` | `W6` | `B7` | `W7` | `·` | `·` | `·` |
| GPU1 | `·` | `F0` | `F1` | `F2` | `F3` | `F4` | `B0` | `F5` | `B1` | `F6` | `B2` | `F7` | `B3` | `W0` | `B4` | `W1` | `B5` | `W2` | `B6` | `W3` | `B7` | `W4` | `W5` | `W6` | `W7` | `·` | `·` |
| GPU2 | `·` | `·` | `F0` | `F1` | `F2` | `B0` | `F3` | `B1` | `F4` | `B2` | `F5` | `B3` | `F6` | `B4` | `F7` | `B5` | `W0` | `B6` | `W1` | `B7` | `W2` | `W3` | `W4` | `W5` | `W6` | `W7` | `·` |
| GPU3 | `·` | `·` | `·` | `F0` | `B0` | `F1` | `B1` | `F2` | `B2` | `F3` | `B3` | `F4` | `B4` | `F5` | `B5` | `F6` | `B6` | `F7` | `B7` | `W0` | `W1` | `W2` | `W3` | `W4` | `W5` | `W6` | `W7` |

两种编排中，每张卡均执行 8 次 F、8 次 B 和 8 次 W，计算工作量一致。ZB 通过拆分 B/W 减少依赖等待；本例仍有流水线启动和收尾空档。该计算编排不包含 optimizer step，实际训练收益还取决于通信、同步和各 stage 的真实计算代价。

### 气泡百分比的统计口径

按 [Megatron 论文 §2.2.1](https://arxiv.org/html/2104.04473v5#S2.SS2.SSS1)的定义，普通非交错 1F1B 在各 stage 计算负载均衡、忽略通信等额外开销的模型下：

```text
p       = pipeline parallel size
m       = global_batch_size / (micro_batch_size × data_parallel_size)
t_b     = 完整 backward 的时间，即拆分后的 t_B + t_W
t_ideal = m × (t_F + t_b)
t_bubble = (p - 1) × (t_F + t_b)

气泡 / 理想计算时间 = t_bubble / t_ideal = (p - 1) / m
气泡 / 总时间     = t_bubble / (t_ideal + t_bubble) = (p - 1) / (m + p - 1)
```

以上图的 `PP=4`、`DP=1`、`GBS=32`、`MBS=4` 为例，`m=8`，普通 1F1B 的两种口径分别为 `37.5%` 和约 `27.3%`。上图 ZB 编排对应的两种比例分别为 `12.5%` 和约 `11.1%`。这些比例来自等代价、零通信开销的计算编排；比较实测结果时也应保持分母一致，并说明统计范围。

## 初始化和接入流程

在已有 Megatron 训练入口中，先导入 BR adaptor，再导入训练接口和模型构造代码：

```python
import megatron_supa

# 随后导入 Megatron 的训练入口和模型构造代码。
```

导入后，适配层会自动注册 ZB 训练参数，并为 local Linear 接入 B/W 拆分实现。在训练命令中添加 `--enable-zero-bubble`，同时使用 `--transformer-impl local`，即可启用 ZB 调度。

训练初始化时会检查 PP、virtual pipeline、梯度累积融合及梯度同步等配置是否满足要求。所需参数见[相关参数](#相关参数)，完整的参数组合见[配置示例](#配置示例)。

## 运行时执行流程

### 调度生成

框架根据 PP stage 数和每轮 microbatch 数，为各 stage 生成 F/B/W 编排，并在配置相同时复用。编排遵循前向数据和反向梯度的依赖关系，将可延后的 W 安排在后续位置。

当前排程假设 F、B、W 等代价，并忽略通信耗时，不会根据设备上的实测耗时自动调整。因此，各 stage 负载不均衡或通信开销较大时，实际执行中的空档可能多于前面的编排示意。

### 单轮训练流程

每轮训练包含以下工作，其中不同 microbatch 的 F/B/W 按编排交错执行，同一 microbatch 保持 F → B → W 的依赖顺序：

1. **前向计算（F）**：接收上游输入，完成本 stage 的前向计算，并向下游传递输出。
2. **输入梯度计算（B）**：根据下游梯度计算输入梯度，向上游传递；将可延后的权重梯度计算留给 W。
3. **权重梯度计算（W）**：在调度指定的位置执行延后的计算，并累积到本轮参数梯度中。
4. **梯度收尾**：完成所有 microbatch 的计算及延后的 W，执行所需的梯度同步和收尾工作。
5. **参数更新**：继续执行训练循环中的 optimizer step，更新模型参数。

### 通信与等待

相邻 stage 在前向阶段传递激活，在反向阶段传递输入梯度，BR 环境中由 BCCL 执行通信。当前 ZB 不支持开启 PP 通信 overlap，收发和等待仍会影响后续计算的推进。

通信调用的耗时可能包含等待对端产生数据、数据传输及同步。分析训练瓶颈时，应结合各 stage 的计算和通信记录判断等待来源。计算编排中的气泡减少，并不保证实际迭代耗时按相同比例缩短。

## 相关参数

### ZB 参数和必要配置

| 参数 | 设置要求 | 作用 |
| --- | --- | --- |
| `--enable-zero-bubble` | 显式添加；默认关闭 | 启用 ZB 调度。 |
| `--pipeline-model-parallel-size` | 大于 `1` | 指定 PP stage 数。 |
| `--transformer-impl` | 设置为 `local` | 使用支持 B/W 拆分的 Linear 实现。 |
| `--micro-batch-size` / `--global-batch-size` | GBS 必须是 MBS × DP size 的整数倍 | 决定每轮 microbatch 数：`m = GBS / (MBS × DP size)`，从而影响编排和气泡。 |
| `--no-gradient-accumulation-fusion` | 不添加，保留默认开启的梯度累积融合 | 满足 ZB 权重梯度计算的要求。 |
| `--zero-bubble-max-pending-backward` | 保持默认值 `auto` | 当前设置不影响调度，也不能用于限制待执行计算数量或峰值显存。 |

### 与其他 overlap 功能的关系

- **DP 梯度通信 overlap**：保持关闭，不添加 `--overlap-grad-reduce`。
- **PP 点对点通信 overlap**：保持关闭；可使用 `--no-overlap-p2p-communication` 显式关闭。
- **TP 通信 overlap**：`--tp-comm-overlap` 面向 Transformer Engine Linear；当前 ZB 使用 local Linear，保持该选项关闭。相关说明见 [TP 通信 overlap 参数说明](tp_comm_overlap.md)。

## 配置示例

### 在已有训练命令中开启 ZB

在训练入口先导入 `megatron_supa`，并将以下参数合并到已经可以运行的训练命令中：

```bash
--enable-zero-bubble \
--pipeline-model-parallel-size 4 \
--tensor-model-parallel-size 1 \
--transformer-impl local \
--micro-batch-size 4 \
--global-batch-size 32 \
--eval-iters 0
```

这个参数片段假设 `WORLD_SIZE=4`、`DP=1`，因此每轮有 8 个 microbatch，与前面的编排对比一致。模型层数、数据、tokenizer、精度和 attention backend 等设置继续由完整训练命令提供；同时保持 gradient accumulation fusion 开启，并关闭下节列出的不支持组合。

已有的 [Llama3 ZB smoke 脚本](../../../examples/llama3/pretrain_llama3_lite_smoke_zb.sh)提供了 local Linear 配合 Flash Attention 的示例：通过自定义 layer spec，仅将无参数的 core attention 替换为 `TEDotProductAttention`，QKV、输出投影和 MLP 仍保留 local 实现。该示例不能扩展解释为 ZB 支持整个 `transformer_impl=transformer_engine` 路径；其数据和环境路径也需按运行机器配置。

## 当前限制和排查建议

### 支持范围

| 项目 | 使用要求 |
| --- | --- |
| Pipeline parallelism | PP size 大于 `1`，使用非交错流水线；不启用 virtual pipeline 或 model chunks。 |
| 模型实现 | 使用 `--transformer-impl local`。 |
| 梯度累积融合 | 保留默认开启状态，不添加 `--no-gradient-accumulation-fusion`。 |
| DP / PP 通信 overlap | 保持关闭，具体设置见[相关参数](#相关参数)。 |
| Embedding 权重梯度延迟 | 不添加 `--defer-embedding-wgrad-compute`。 |
| 自定义通信 shape 调整 | 不支持自定义调整 stage 间通信 shape 的回调。 |
| 验证、测试及其他仅前向任务 | 使用普通调度单独执行。 |
| Megatron FSDP | 当前不支持与 ZB 的 B/W 拆分组合使用。 |

变长序列、MTP 及自定义模型需要单独验证相邻 stage 之间张量结构、shape 和 dtype 的兼容性。

ZB 训练任务中应关闭内置验证和测试。使用标准 Megatron 训练入口时，设置 `--eval-iters 0`，并保持 `--full-validation` 关闭；需要评估时，使用未开启 `--enable-zero-bubble` 的独立任务。

### 启动检查

启动训练时检查以下配置：

1. 训练入口先导入 `megatron_supa`，再导入训练接口和模型构造代码。
2. 实际启动命令包含 `--enable-zero-bubble`；启动日志中的 `enable_zero_bubble` 为 `True`，`transformer_impl` 为 `local`，PP size 大于 `1`。
3. 未启用 virtual pipeline、DP / PP 通信 overlap 和内置评估，并保留梯度累积融合。

### 常见问题

| 现象 | 排查建议 |
| --- | --- |
| 开启后仍有气泡或提速不明显 | 在相同模型、精度、batch 和并行配置下比较稳定迭代；检查各 stage 的计算负载、microbatch 数量，以及通信和同步等待。前面的编排图仅表示计算依赖，不能直接作为实测性能结论。 |
| W 集中在迭代后半段 | 延后 W 是调度的一部分。结合上游是否能够更早推进、迭代总耗时及峰值显存判断效果。 |
| 修改 `--zero-bubble-max-pending-backward` 没有效果 | 保持默认值 `auto`；该参数目前不影响编排，也不能限制峰值显存。 |
| 通信长时间等待或训练不再推进 | 收集所有 rank 的日志，先检查是否有 rank 提前报错，再对照相邻 stage 最后完成的计算和收发记录；使用自定义模型时，检查收发张量的 shape 和 dtype 是否匹配。通信调用的耗时包含等待对端和同步，不能全部视为数据传输耗时。 |
| 进入验证或测试时报错 | 关闭 ZB 训练任务中的内置评估，使用普通调度单独执行评估任务。 |
| 出现权重梯度任务为空或未完成的错误 | 核对 local 模型实现、梯度累积融合和并行配置是否符合上述要求；排查时保留完整启动参数、所有 rank 的日志及报错堆栈。 |

## 参考资料

- [Zero Bubble Pipeline Parallelism](https://arxiv.org/abs/2401.10241)
- [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473)
