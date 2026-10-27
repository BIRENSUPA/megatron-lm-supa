<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Distributed Optimizer

本文只覆盖通过 `--use-distributed-optimizer` 使能的普通 Megatron distributed optimizer。Megatron FSDP、Torch FSDP2 以及 `--data-parallel-sharding-strategy` 对应的 FSDP 模式不在本文范围内。

## 作用和适用场景

Distributed optimizer 的核心作用是节省显存：它把 optimizer 持有的状态按 data-parallel rank 分片，而不是在每个 data-parallel rank 上完整复制一份 optimizer state。它最适合 optimizer state、main parameter buffer、main gradient buffer 占用显存较多的场景，尤其是使用 Adam 训练大模型并且 data parallel size 不小的时候。

Distributed optimizer 是一种 data-parallel 维度上的 sharding 机制，可以和 tensor parallelism、pipeline parallelism、context parallelism、expert parallelism 一起使用。Tensor-parallel 和 pipeline-parallel 切分仍然属于 model-parallel 维度；distributed optimizer 只对当前 model-parallel rank 本地持有的参数做 data-parallel sharding。

Pipeline interleaving 与 distributed optimizer 兼容；`--overlap-param-gather-with-optimizer-step` 还专门要求使用 interleaved pipeline parallelism。

## 显存节省原理

不使用 distributed optimizer 时，每个 data-parallel rank 都会保存完整的模型参数、梯度、fp32 main parameters 以及 Adam states。使用 distributed optimizer 后，每个 rank 仍然需要保留一份用于 forward/backward 的完整低精度模型参数 buffer，但 optimizer step 只持有 main gradients、main parameters 和 optimizer state 的本地分片。

设 data-parallel size 为 `d`，每个参数对应的理论 optimizer/model-state 显存开销为：

| 参数 / 梯度 dtype | 非 distributed optimizer | Distributed optimizer |
| --- | ---: | ---: |
| `fp16` parameters, `fp16` gradients | 20 bytes | `4 + 16 / d` bytes |
| `bf16` parameters, `fp32` gradients | 18 bytes | `6 + 12 / d` bytes |
| `fp32` parameters, `fp32` gradients | 16 bytes | `8 + 8 / d` bytes |

其中不能被 sharding 的部分是 forward/backward 所需的模型侧参数和梯度存储；可以被 sharding 的部分是 optimizer 持有的状态。例如 bf16 parameters + fp32 main gradients 场景下，本地仍保留 bf16 模型参数和 fp32 梯度累积 buffer，而 fp32 main parameters 和 Adam moments 会按 data-parallel rank 分片。

### Weight、gradient 和 optimizer state 的 dtype

一般训练时常用的 baseline 配置是只开启 `--use-distributed-optimizer --bf16`，不额外指定 precision-aware optimizer 或 bf16 gradient reduction。此时默认存储形态通常是：

| 存储对象 | 保存范围 | dtype |
| --- | --- | --- |
| 模型参数 / parameter buffer | 每个 DP rank 完整保存一份 | `bf16` |
| 模型梯度 / grad buffer | 每个 DP rank 完整保存一份 | `fp32` |
| main parameters | 按 DP rank 分片 | `fp32` |
| main gradients | 按 DP rank 分片；为完整 fp32 模型 grad buffer 中本 rank shard 的 slice/view，通常不额外占用一份 shard 显存 | `fp32` |
| Adam `exp_avg` | 按 DP rank 分片 | `fp32` |
| Adam `exp_avg_sq` | 按 DP rank 分片 | `fp32` |

这是因为 `--bf16` 会让模型参数使用 bf16；同时在未指定 `--grad-reduce-in-bf16` 且 main gradients 仍为 fp32 时，Megatron 会默认启用 fp32 gradient accumulation/reduction。因此它对应前面显存表中的 `bf16 parameters, fp32 gradients` 场景：每个 DP rank 仍持有完整 bf16 model parameters 和完整 fp32 grad buffer，但 optimizer 侧的 fp32 main params、main grads 和 Adam states 只保存本 rank 的 shard。

更一般地，Megatron mixed precision 训练里通常会同时出现几类不同用途的 weight / gradient 存储：

| 存储对象 | 是否每个 DP rank 完整保存 | 常见 dtype | 主要用途 / 相关配置 |
| --- | --- | --- | --- |
| 模型参数 / parameter buffer | 是 | `fp16`、`bf16` 或 `fp32` | forward/backward 使用的参数。由 `--fp16`、`--bf16` 影响；未启用 mixed precision 时通常是 fp32。 |
| 模型梯度 / grad buffer | 是 | 默认跟参数 dtype 相同；也可能是 `fp32` | backward 写入和 DP 梯度通信使用的连续 buffer。`--accumulate-allreduce-grads-in-fp32` 会让梯度累积和 all-reduce/reduce-scatter 使用 fp32；bf16 训练下如果 `--main-grads-dtype fp32` 且未指定 `--grad-reduce-in-bf16`，Megatron 会默认打开 fp32 gradient accumulation/reduction。 |
| main parameters | 非 distributed optimizer 下完整保存；distributed optimizer 下按 DP rank 分片 | 通常 `fp32` | optimizer step 使用的 master weights。precision-aware optimizer 下可通过 `--main-params-dtype {fp32,fp16}` 改变存储 dtype。 |
| main gradients | 非 distributed optimizer 下通常完整保存；distributed optimizer 下按 DP rank 分片 | 通常 `fp32` | optimizer step 使用的梯度。precision-aware optimizer 下可通过 `--main-grads-dtype {fp32,bf16}` 控制。 |
| Adam states (`exp_avg` / `exp_avg_sq`) | 非 distributed optimizer 下完整保存；distributed optimizer 下按 DP rank 分片 | 通常 `fp32` | Adam 一阶、二阶矩。precision-aware optimizer 下可通过 `--exp-avg-dtype`、`--exp-avg-sq-dtype` 改变存储 dtype。 |

因此，fp16/bf16 训练一般会保留 fp32 main parameters 和 fp32 Adam states，但不等价于每个 rank 都长期保留一份完整 fp32 master gradients。普通 distributed optimizer 只需要 optimizer step 当前 rank 负责的 main-gradient shard；完整模型侧 grad buffer 是否使用 fp32，取决于 DDP gradient buffer / reduction 配置。

几个容易混淆的配置关系：

- `--fp16` / `--bf16`：主要决定模型参数和计算侧 parameter buffer dtype。
- `--accumulate-allreduce-grads-in-fp32`：让 gradient accumulation 以及 DP gradient reduction 使用 fp32，对应 DDP 中的 `grad_reduce_in_fp32`。
- `--grad-reduce-in-bf16`：bf16 训练下强制用 bf16 做 gradient reduction，会关闭默认的 fp32 gradient accumulation/reduction。
- `--main-grads-dtype`、`--main-params-dtype`、`--exp-avg-dtype`、`--exp-avg-sq-dtype`：precision-aware optimizer 下控制 optimizer 侧 main gradients、main parameters 和 Adam states 的存储 dtype。
- `--ddp-param-name-patterns-for-fp32-local-accumulation`：只对匹配参数名的参数使用 fp32 本地梯度累积；不应和全局 fp32 gradient reduction 混用。

## Build 和初始化流程

高层初始化路径如下：

1. `training.py` 根据训练参数构造 `DistributedDataParallelConfig`。
2. `get_model(..., wrap_with_ddp=True)` 将每个 model chunk 包装为 Megatron `DistributedDataParallel`。
3. DDP 按 parameter dtype 和 gradient dtype 对可训练参数分组，并分配 `_ParamAndGradBuffer`。
4. 每个 `_ParamAndGradBuffer` 创建连续的 parameter/gradient storage 以及通信 buckets。
5. `setup_model_and_optimizer()` 构造 `OptimizerConfig`，并调用 `get_megatron_optimizer()`。
6. `get_megatron_optimizer()` 分离 dense 和 expert parameter groups；当 `config.use_distributed_optimizer` 为 true 时选择 `DistributedOptimizer`。
7. `DistributedOptimizer.__init__()` 建立 parameter 到 grad-buffer shard range 的映射，按 data-parallel rank 切分 buckets，创建本地 main-parameter / optimizer-state shards，并将底层 optimizer 的 param groups 替换成本地 shards。

关键代码位置：

- `third-party/megatron-lm/megatron/core/optimizer/distrib_optimizer.py`：`DistributedOptimizer` 实现。
- `third-party/megatron-lm/megatron/core/optimizer/__init__.py`：optimizer factory，以及 dense/expert optimizer 构造逻辑。
- `third-party/megatron-lm/megatron/core/distributed/distributed_data_parallel.py`：DDP 连续 buffer 和 bucket 分配。
- `third-party/megatron-lm/megatron/core/distributed/param_and_grad_buffer.py`：parameter/gradient buffer 和 bucket 机制。
- `third-party/megatron-lm/megatron/training/training.py`：DDP config 构造、model wrapping、optimizer setup。
- `third-party/megatron-lm/megatron/training/arguments.py`：CLI 参数定义和校验。

## 运行时更新流程

Megatron 的 distributed optimizer 实现使用连续的 parameter buffer 和 gradient buffer。核心更新流程如下：

1. Backward 将模型梯度写入连续的 grad buffers。
2. 使能 distributed optimizer 后，data-parallel 梯度通信使用 reduce-scatter，而不是 all-reduce。
3. 每个 data-parallel rank 只收到自己负责的、已经 reduce 完成的梯度分片。
4. 本地 optimizer step 只更新该 rank 持有的 main parameters 和 optimizer state 分片。
5. 更新后的参数分片被复制并按需 cast 回模型 parameter buffer 的对应分片。
6. Data-parallel all-gather 将完整的、更新后的模型 parameter buffer 重新组装到每个 data-parallel rank 上，供下一轮 forward 使用。

使用 `--overlap-grad-reduce` 时，reduce-scatter 可以在 backward 过程中按 bucket 异步发起。使用 `--overlap-param-gather` 时，parameter all-gather 可以和 forward prefetch 重叠。

## 高级功能说明

### `--ddp-reduce-scatter-with-fp32-accumulation`

这个开关用于 distributed optimizer 的 reduce-scatter 路径。它适合在以下场景使用：模型梯度 buffer / 通信数据希望保持低精度，以减少通信带宽和 buffer 开销；但又希望跨 DP rank 的梯度求和用 fp32 累加，以降低 fp16/bf16 reduction 的数值误差。

它解决的是“reduce-scatter 的累加精度”问题，而不是“optimizer 有没有 fp32 main gradients”问题。如果普通 reduce-scatter 已经在低精度 collective 中完成了求和，后续再把结果 cast 成 fp32 main gradients 也无法恢复低精度累加造成的误差。

实现逻辑上，它把 reduce-scatter 拆成两步：

1. 用 `all_to_all_single` 在 DP group 内交换 shard，让每个 rank 收到自己负责的 gradient shard 在所有 DP ranks 上的贡献。
2. 每个 rank 对收到的这些 shard 做本地 `torch.sum(..., dtype=torch.float32)`，得到 fp32 累加结果，再写回 reduce-scatter 的 output shard。

例如 DP size = 4，完整 gradient buffer 按 shard 切成 `[A, B, C, D]`。普通 reduce-scatter 会让 rank0 得到 `A0 + A1 + A2 + A3`，rank1 得到 `B0 + B1 + B2 + B3`。这个实现先通过 all-to-all 让 rank0 收到 `[A0, A1, A2, A3]`，再在 rank0 本地用 fp32 求和；其他 rank 同理。

这个路径的代价是需要额外的临时 all-to-all output buffer，并且 reduce 变成本地 fp32 sum。当前 Megatron 实现还限制它不支持 `--num-distributed-optimizer-instances > 1`，异步 overlap 场景下只支持单个 bucket。因此它通常只在低精度 reduce-scatter 数值稳定性成为问题、但又不想把整条 gradient reduction 路径都切到 fp32 时使用。

相关代码位置：

- `third-party/megatron-lm/megatron/core/distributed/reduce_scatter_with_fp32_accumulation.py`：用 all-to-all + 本地 fp32 sum 实现 reduce-scatter 语义。
- `third-party/megatron-lm/megatron/core/distributed/param_and_grad_buffer.py`：当 `reduce_scatter_with_fp32_accumulation` 为 true 时替换默认 reduce-scatter 实现，并检查多 optimizer instances / bucket 限制。
- `third-party/megatron-lm/megatron/core/distributed/distributed_data_parallel.py`：根据 `grad_reduce_in_fp32` 决定 grad buffer dtype。
- `third-party/megatron-lm/megatron/training/arguments.py`：定义 `--accumulate-allreduce-grads-in-fp32`、`--grad-reduce-in-bf16`、`--ddp-reduce-scatter-with-fp32-accumulation` 和 precision-aware optimizer dtype 参数。

### `--overlap-param-gather-with-optimizer-step`

普通 distributed optimizer 的参数更新流程是：backward 写入 grad buffer，data-parallel reduce-scatter 得到本地梯度分片，optimizer step 更新本地 main parameters 和 optimizer state 分片，然后 parameter all-gather 把更新后的模型参数重新组装到每个 data-parallel rank 上。

`--overlap-param-gather` 可以把 parameter all-gather 与下一轮 forward 的 prefetch 重叠，但第一批 forward 需要的参数仍然容易卡在 optimizer step 之后、下一轮 forward 之前。`--overlap-param-gather-with-optimizer-step` 的作用就是把这第一批 parameter all-gather 提前到 optimizer step 过程中发起，减少 optimizer step 到下一轮 forward 之间等待参数同步的时间。

这个开关要求 interleaved pipeline parallelism，是因为 interleaved schedule 有更细粒度的 model chunk 执行顺序。每个 physical pipeline rank 上有多个 virtual pipeline chunks，下一轮 forward 会按 chunk 顺序逐步使用参数，因此 Megatron 可以只提前 gather 最先需要的 chunk 参数，而不是在 optimizer step 阶段集中 all-gather 所有参数。这样更容易把通信放进已有调度空隙里，也避免提前 gather 过多参数带来的通信和显存压力。

因此，`--overlap-param-gather-with-optimizer-step` 不是 distributed optimizer 的基础开关，而是更激进的 overlap 优化。通常只有在已经使用 `--use-distributed-optimizer`、`--overlap-grad-reduce`、`--overlap-param-gather`，并且训练本身采用 interleaved pipeline parallelism 时才考虑启用。

### `--num-distributed-optimizer-instances`

`--num-distributed-optimizer-instances` 会把完整 data-parallel domain 拆成多个 distributed optimizer instances。它不是单纯的 bucket 调优，而是改变 optimizer state 的 sharding domain：optimizer state/main params 在每个 instance 内分片，在不同 instances 之间复制。

例如 DP size = 8、`--num-distributed-optimizer-instances 2`：

```text
完整 DP domain:      [0 1 2 3 | 4 5 6 7]
optimizer instance:  [0 1 2 3] 和 [4 5 6 7]
```

这时每个 instance 内按 4 个 rank 分片 optimizer state，而不是按完整 8 个 rank 分片。因此它会带来几个直接影响：

- 显存占用上升：optimizer state/main params 从按 8 份分片变成按 4 份分片，并在两个 instances 之间复制。
- 梯度语义仍保持完整 DP：先在每个 instance 内 reduce-scatter，再在 instances 之间对同一 shard 做 all-reduce，最终 optimizer step 使用的仍是完整 DP domain 上聚合后的梯度。
- parameter all-gather 范围变小：optimizer step 后，每个 instance 只需要在自己的 4 个 rank 内 all-gather updated parameter shards。
- 通信拓扑更分层：大 DP group 的 optimizer-domain 通信被拆成 intra-instance 通信和 inter-instance shard 同步，适合 DP 很大、跨节点拓扑复杂、全局 collective 不稳定或效率较差的场景。

相关代码位置：

- `third-party/megatron-lm/megatron/core/parallel_state.py`：根据 `num_distributed_optimizer_instances` 构造 intra/inter distributed optimizer process groups。
- `third-party/megatron-lm/megatron/core/distributed/distributed_data_parallel.py`：DDP 选择 intra DP/CP group 分配 parameter/gradient buffers，并在多 instances 时设置 `inter_dist_opt_group`。
- `third-party/megatron-lm/megatron/core/distributed/param_and_grad_buffer.py`：distributed optimizer 梯度同步先在 `intra_distributed_optimizer_instance_group` 上 reduce-scatter；当 `num_distributed_optimizer_instances > 1` 时，再通过 `inter_distributed_optimizer_instance_group` 对本地 shard 做 all-reduce。

这个参数适合用更多 optimizer state 显存换更小的 optimizer sharding domain 和更可控的通信拓扑。如果 DP 规模不大，或者默认 full-DP distributed optimizer 的通信已经稳定高效，一般不需要启用多个 instances。

## 相关参数

### 核心开关

| 参数 | 范围 | 作用 / 使用场景 |
| --- | --- | --- |
| `--use-distributed-optimizer` | 普通 distributed optimizer | 使能 `DistributedOptimizer`；梯度使用 reduce-scatter，optimizer state/main params 按 data-parallel rank 分片。 |
| `--num-distributed-optimizer-instances` | 普通 distributed optimizer | 将 data-parallel domain 拆成多个 distributed-optimizer instances。这是一种 hybrid sharding：optimizer state 在每个 instance 内分片，在 instances 之间复制。适用于不希望整个 DP 维度都参与 optimizer sharding、需要折中通信和显存的场景。 |

### 通信 overlap 和 bucket

| 参数 | 范围 | 作用 / 使用场景 |
| --- | --- | --- |
| `--overlap-grad-reduce` | DDP + distributed optimizer | 通过通信 buckets 将梯度 reduce-scatter/all-reduce 与 backward 重叠。对 distributed optimizer 来说，具体是 overlap reduce-scatter。 |
| `--overlap-param-gather` | distributed optimizer、Megatron FSDP 或 `dist_muon` | 将 parameter all-gather 与 forward 重叠。普通 distributed optimizer 路径下要求同时使用 `--use-distributed-optimizer`、`--overlap-grad-reduce`。 |
| `--overlap-param-gather-with-optimizer-step` | 普通 distributed optimizer | 将第一个 parameter all-gather 与 optimizer step 重叠。要求 `--use-distributed-optimizer`、`--overlap-param-gather`、interleaved pipeline parallelism，且当前不支持 distributed checkpointing。 |
| `--ddp-bucket-size` | DDP | data-parallel 通信 bucket 的最大元素数。未设置时默认使用 `max(40000000, 1000000 * dp_size)`。当 `--overlap-grad-reduce` 为 false 时会禁用 bucket。 |
| `--ddp-num-buckets` | DDP | 通过 bucket 数量控制 data-parallel 通信切分。它与 `--ddp-bucket-size` 互斥。 |
| `--ddp-pad-buckets-for-high-nccl-busbw` | DDP | 对 bucket size 做 padding，以提升大 DP size 下的 NCCL bus bandwidth。 |
| `--ddp-average-in-collective` | DDP | 在 collective 内直接完成梯度平均，而不是在 collective 前先除以 DP size。 |
| `--create-all-gather-group` | distributed communication | 创建单独的 all-gather process group，使 reduce-scatter 和 all-gather 有机会重叠。 |
| `--use-nccl-ub` | DP/FSDP communication | 对 DP/FSDP communication buffers 使用 NCCL user-buffer registration，降低 overlap 通信时的 SM 使用量。它不是 FSDP-only。 |

### 梯度累积 / 通信精度

| 参数 | 范围 | 作用 / 使用场景 |
| --- | --- | --- |
| `--ddp-reduce-scatter-with-fp32-accumulation` | DDP reduce-scatter | 通信链路上传输低精度数据，但在本地使用 fp32 做 accumulation。 |
| `--ddp-param-name-patterns-for-fp32-local-accumulation` | DDP | 对匹配参数名的参数使用 fp32 本地梯度累积。pattern `all` 匹配全部参数。概念上不要和已经全局使用 fp32 gradient reduction 的配置混用。 |
| `--fp8-param-gather` | distributed optimizer、FSDP 或 inference | 在支持的情况下使用 fp8 做 parameter all-gather。普通 distributed optimizer 路径下可降低 param-gather 带宽，但受 FP8 training 相关约束限制。 |

### Precision-aware optimizer 显存参数

这些参数不是 FSDP-specific。它们要求使用 `--use-precision-aware-optimizer`，而该参数又要求 `--use-distributed-optimizer`，并且要求 Adam；常规 GPU 路径需要支持相关 dtype 设置的 TransformerEngine FusedAdam。

| 参数 | 作用 / 使用场景 |
| --- | --- |
| `--use-precision-aware-optimizer` | 允许 main params 和 Adam states 使用更低精度存储。 |
| `--main-grads-dtype {fp32,bf16}` | main gradients 的 dtype，默认 fp32。 |
| `--main-params-dtype {fp32,fp16}` | main parameters 的 dtype，默认 fp32。 |
| `--exp-avg-dtype {fp32,fp16,bf16,fp8}` | Adam 一阶矩的存储 dtype，不一定改变 kernel compute precision。 |
| `--exp-avg-sq-dtype {fp32,fp16,bf16,fp8}` | Adam 二阶矩的存储 dtype，不一定改变 kernel compute precision。 |

### Checkpoint 相关参数

| 参数 | 作用 / 使用场景 |
| --- | --- |
| `--ckpt-format` | 选择 checkpoint 格式；使用 `torch_dist` 等 distributed checkpoint 格式便于灵活 resume。`--use-dist-ckpt` 已废弃且不生效。若禁用 Gloo process groups，distributed optimizer 要求使用 distributed checkpointing。 |
| `--dist-ckpt-optim-fully-reshardable` | 以支持跨不同 parallel layout 重新 sharding 的格式保存 optimizer state。 |
| `--distrib-optim-fully-reshardable-mem-efficient` | fully reshardable distributed optimizer checkpointing 的省内存变体；该路径要求启用 Gloo process groups。 |

## 实用建议

普通 distributed optimizer 的常见 baseline 配置如下：

```bash
--use-distributed-optimizer \
--overlap-grad-reduce \
--overlap-param-gather
```

只有在 interleaved pipeline parallel 训练且满足额外调度约束时，才使用 `--overlap-param-gather-with-optimizer-step`。只有确认已经使能通信 overlap，并且默认 bucket size 不适合当前集群规模时，才进一步调优 `--ddp-bucket-size` 或 `--ddp-num-buckets`。
