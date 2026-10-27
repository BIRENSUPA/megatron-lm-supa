# TP 通信 overlap 参数说明

本文梳理 Megatron-LM BR adaptor 当前随仓库引入的 Megatron-Core 中，与 Tensor Parallel（TP）通信 overlap 相关的参数。这里的 TP 通信 overlap 主要指在 TP 线性层中，将 GEMM 与 AllGather / ReduceScatter 等 collective 通信重叠执行。

## 快速结论

开启 TP 通信 overlap 的主开关是 `--tp-comm-overlap`。代码要求该功能只能在 sequence parallel 开启时使用：`third-party/megatron-lm/megatron/training/arguments.py` 检查 `args.tp_comm_overlap` 后断言 `args.sequence_parallel == True`；YAML 参数路径也在 `third-party/megatron-lm/megatron/training/yaml_arguments.py` 做同样检查。

该功能依赖 Transformer Engine 的 user buffer（UB）机制。初始化阶段如果 `args.tp_comm_overlap` 为真，会调用 `_initialize_tp_communicators()`：`third-party/megatron-lm/megatron/training/initialize.py`。初始化函数会读取 `--tp-comm-overlap-cfg` 指定的 YAML，并调用 `transformer_engine.pytorch.module.base.initialize_ub(...)` 创建 UB：`third-party/megatron-lm/megatron/training/initialize.py`。

## 参数总览

| 参数 | 默认值 | overlap 粒度 | 通信阶段 | 作用范围 | 作用 |
| --- | --- | --- | --- | --- | --- |
| `--tp-comm-overlap` | `False` | 总开关 | 前向 + 反向 | 训练 TP 线性层 | 总开关，允许 Linear 层执行与 TP AllGather / ReduceScatter 通信重叠 |
| `--tp-comm-overlap-cfg` | `None` | 配置文件 | 前向 + 反向 | UB 初始化 | 指定 TP overlap 的 UB YAML 配置文件 |
| `--tp-comm-overlap-ag` | `True` | chunk / split 级 | 前向为主 | TE Linear / LayerNormLinear | `--tp-comm-overlap` 的子开关，控制 AllGather 与 GEMM 的 pipeline overlap |
| `--tp-comm-overlap-rs` | `True` | chunk / split 级 | 前向 + 反向 | TE Linear | `--tp-comm-overlap` 的子开关，控制 ReduceScatter 与 GEMM 的 pipeline overlap |
| `--tp-comm-overlap-rs-dgrad` | `False` | chunk / split 级 | 反向 | TE column-parallel Linear / LayerNormLinear | `--tp-comm-overlap` 的子开关，控制 activation-gradient GEMM 与 ReduceScatter 的 pipeline overlap |
| `--tp-comm-bulk-wgrad` | `True` | 整块级 | 反向 | TE column-parallel Linear / LayerNormLinear | 按 TE 实现，控制 ReduceScatter 与 GEMM 的 bulk overlap：ReduceScatter 对象是 activation-gradient GEMM 产出的 dgrad，GEMM 是 weight-gradient 计算的 GEMM |
| `--tp-comm-bulk-dgrad` | `True` | 整块级 | 反向 | TE column-parallel Linear / LayerNormLinear | 按 TE 实现，控制 AllGather 与 GEMM 的 bulk overlap：AllGather 对象是拼接后用于 weight-gradient 计算的前向 input activation，GEMM 是 activation-gradient 计算的 GEMM |
| `tp_comm_overlap_disable_qkv` | `False` | 局部禁用 | 前向 + 反向 | QKV LayerNormLinear | 禁用 QKV 层的 AllGather -> GEMM overlap；当前无 CLI 参数，只能通过配置对象或代码路径设置 `config.tp_comm_overlap_disable_qkv=True` |
| `tp_comm_overlap_disable_fc1` | `False` | 局部禁用 | 前向 + 反向 | MLP FC1 LayerNormLinear | 禁用 FC1 层的 AllGather -> GEMM overlap；当前无 CLI 参数，只能通过配置对象或代码路径设置 `config.tp_comm_overlap_disable_fc1=True` |
| `--tp-comm-split-ag` | `True` | chunk / split 级 | 前向 | 旧版 TE 兼容 | 已在 TE v1.6.0 后废弃；旧路径中控制 AllGather split overlap |
| `tp_comm_atomic_ag` | `False` | atomic 级 | 前向 | 旧版 TE 兼容 | 已在 TE v1.6.0 后废弃；旧路径中控制 AllGather atomic GEMM overlap；当前无 CLI 参数 |
| `--tp-comm-split-rs` | `True` | chunk / split 级 | 前向 | 旧版 TE 兼容 | 已在 TE v1.6.0 后废弃；旧路径中控制 ReduceScatter split overlap |
| `tp_comm_atomic_rs` | `False` | atomic 级 | 前向 | 旧版 TE 兼容 | 已在 TE v1.6.0 后废弃；旧路径中控制 ReduceScatter atomic GEMM overlap；当前无 CLI 参数 |
| `--tp-comm-bootstrap-backend` | `nccl` | 初始化选项 | 初始化 | UB 初始化 | 设置 TP overlap communicator bootstrap backend，可选 `nccl` / `mpi` / `gloo` |
| `--decoder-tp-comm-overlap` | `False` | 总开关 | 前向 + 反向 | VLM decoder | 只在 VLM decoder / LLM backbone 启用 TP 通信与 GEMM overlap |
| `--inference-fuse-tp-communication` | `False` | fused kernel | 推理前向 | 推理 optimized transformer | 推理路径使用 fused reduce-scatter + residual/norm + all-gather kernel；不属于训练 UB overlap，但同属 TP 通信融合/重叠相关开关 |

> 注：多数 TP overlap CLI 参数名由 dataclass 字段自动生成；`tp_comm_overlap_disable_qkv`、`tp_comm_overlap_disable_fc1`、`tp_comm_atomic_ag`、`tp_comm_atomic_rs` 在 `third-party/megatron-lm/megatron/training/arguments.py` 被排除，因此当前没有自动生成的 CLI 参数，只能通过配置对象或代码路径设置。

## 逐项说明与代码依据

### `--tp-comm-overlap`

主开关。`ModelParallelConfig` 对它的定义说明为：开启后，Linear 层执行可以与 TP collective（例如 AllGather / ReduceScatter）重叠，尽可能覆盖 forward 和 backward 中的线性层。依据见 `third-party/megatron-lm/megatron/core/model_parallel_config.py`：

```python
tp_comm_overlap: bool = False
"""If true, allows overlapping of Linear layer execution with tensor parallel communication
   collectives like AllGather/ReduceScatter. Overlapping is done for the linear layers wherever
   possible during the forward and the backward pass.
"""
```

使用约束：必须开启 sequence parallel。普通 argparse 路径见 `third-party/megatron-lm/megatron/training/arguments.py`：

```python
if args.tp_comm_overlap:
    assert args.sequence_parallel == True, 'Tensor parallel communication/GEMM overlap can happen only when sequence parallelism is enabled'
```

YAML 参数路径同样检查，见 `third-party/megatron-lm/megatron/training/yaml_arguments.py`。

初始化逻辑见 `third-party/megatron-lm/megatron/training/initialize.py`：

```python
if args.tp_comm_overlap:
    _initialize_tp_communicators()
```

### `--tp-comm-overlap-cfg`

指定 TP overlap 的 user buffer 配置文件。CLI 定义在 `third-party/megatron-lm/megatron/training/arguments.py`：

```python
group.add_argument('--tp-comm-overlap-cfg', type=str, default=None,
                   help='Config file when tp_comm_overlap is enabled.')
```

初始化时读取 YAML，并传给 Transformer Engine UB 初始化，见 `third-party/megatron-lm/megatron/training/initialize.py`：

```python
if args.tp_comm_overlap_cfg is not None:
    with open(args.tp_comm_overlap_cfg, "r") as stream:
        ub_cfgs = yaml.safe_load(stream)
else:
    ub_cfgs = {}
```

随后作为 `ub_cfgs=ub_cfgs` 传入 `initialize_ub(...)`，见 `third-party/megatron-lm/megatron/training/initialize.py`。

`--tp-comm-overlap-ag`、`--tp-comm-overlap-rs` 和 `--tp-comm-overlap-rs-dgrad` 是 `--tp-comm-overlap` 的细分子开关：总开关不开时，这些参数都不生效；总开关开启后，它们分别控制 AG / RS / activation-gradient RS 方向是否做 pipeline overlap。

### `--tp-comm-overlap-ag`

控制 AllGather 与 GEMM 的 pipeline overlap。定义在 `third-party/megatron-lm/megatron/core/model_parallel_config.py`：

```python
tp_comm_overlap_ag: bool = True
"""Controls All-Gather overlap with GEMM by pipelining the GEMM and All-Gather.
   Don't care if tp_comm_overlap is False.
"""
```

在 TE Linear 中会转成 `ub_overlap_ag`，见 `third-party/megatron-lm/megatron/core/extensions/transformer_engine.py`：

```python
if self.config.tp_comm_overlap and parallel_mode != "duplicated":
    extra_kwargs["ub_overlap_ag"] = self.config.tp_comm_overlap_ag
```

在 TELayerNormColumnParallelLinear 中也会设置 `ub_overlap_ag`，并可能被 QKV / FC1 的局部禁用开关覆盖，见 `third-party/megatron-lm/megatron/core/extensions/transformer_engine.py`。

### `--tp-comm-overlap-rs`

控制 ReduceScatter 与 GEMM 的 pipeline overlap。定义在 `third-party/megatron-lm/megatron/core/model_parallel_config.py`：

```python
tp_comm_overlap_rs: bool = True
"""Controls Reduce-Scatter overlap with GEMM by pipelining the GEMM and Reduce-Scatter.
   Don't care if tp_comm_overlap is False.
"""
```

TE Linear 中对应 `ub_overlap_rs`，见 `third-party/megatron-lm/megatron/core/extensions/transformer_engine.py`：

```python
extra_kwargs["ub_overlap_rs"] = self.config.tp_comm_overlap_rs
```

### `--tp-comm-overlap-rs-dgrad`

控制 activation-gradient GEMM 与 ReduceScatter 的 pipeline overlap。这里的 activation-gradient 对应代码命名中的 `dgrad`，即 backward 中计算输入 activation 梯度的 GEMM。定义在 `third-party/megatron-lm/megatron/core/model_parallel_config.py`：

```python
tp_comm_overlap_rs_dgrad: bool = False
"""If true, allows Reduce-Scatter overlap with DGRAD GEMM by pipelining the
   GEMM and Reduce-Scatter splits. Don't care if tp_comm_overlap is False.
"""
```

TELayerNormColumnParallelLinear 中在 TE 版本支持时设置 `ub_overlap_rs_dgrad`，见 `third-party/megatron-lm/megatron/core/extensions/transformer_engine.py`：

```python
extra_kwargs["ub_overlap_rs_dgrad"] = self.config.tp_comm_overlap_rs_dgrad
```

### `--tp-comm-bulk-wgrad`

按当前 TE 实现，作用于 TE column-parallel Linear / LayerNormLinear，控制 ReduceScatter 与 backward weight-gradient GEMM 的 bulk overlap。这里 ReduceScatter 的对象是 activation-gradient GEMM 产出的 dgrad，也就是输入 activation 的梯度；ReduceScatter 后得到当前 TP rank 对应的 activation-gradient 分片。与它并发的 GEMM 是 weight-gradient 计算的 GEMM。这里的 `bulk` 指整块级别的并发 overlap：TE 不把同一次通信/GEMM 细分成多个 `num_splits` chunk 做流水，而是尽量让一整段 TP 通信和另一整段 backward GEMM 在不同执行资源上并发。它和 `pipeline` / `split` 的区别是粒度更粗、调度开销更低，但可隐藏通信的精细程度也更弱。

### `--tp-comm-bulk-dgrad`

按当前 TE 实现，作用于 TE column-parallel Linear / LayerNormLinear，控制 AllGather 与 backward activation-gradient GEMM 的 bulk overlap。这里 AllGather 的对象是前向 input activation，AllGather 后拼接出的完整 input activation 用于后续 weight-gradient 计算；与它并发的 GEMM 是 activation-gradient 计算的 GEMM。这里同样是整块通信与整块 GEMM 的并发调度，不是 `num_splits` 驱动的 chunk pipeline。

> 注意：Megatron 侧 `tp_comm_bulk_wgrad` / `tp_comm_bulk_dgrad` 的注释与当前 TE 实现语义相反；实际行为应以 TE 实现为准。

在 UB YAML 中，bulk 路径通常写成 `method: bulk`，并通过 `num_sm` / `set_sm_margin` 控制通信 kernel 可用资源，而不是通过 `num_splits` 控制 chunk 数。例如 `third-party/megatron-lm/tests/functional_tests/test_cases/gpt/gpt3_weekly_mcore_tp2_pp2_current_scaling_native_fp8_tp_pp_sp_tp_overlap/tp_comm_overlap_cfg.yaml`：

```yaml
fc1_dgrad:
  method: bulk
  num_sm: 2
  set_sm_margin: 0
```

### `tp_comm_overlap_disable_qkv`

禁用 QKV 层上的 AllGather -> GEMM overlap。定义在 `third-party/megatron-lm/megatron/core/model_parallel_config.py`：

```python
tp_comm_overlap_disable_qkv: bool = False
"""
   If true, the AllGather -> Gemm overlap for QKV gets disabled
"""
```

QKV 层通过 `tp_comm_buffer_name='qkv'` 标识，见 `third-party/megatron-lm/megatron/core/transformer/attention.py`。当 buffer name 是 `qkv` 且该开关为真时，代码将 `ub_overlap_ag` 和 `ub_overlap_rs_dgrad` 置为 `False`，见 `third-party/megatron-lm/megatron/core/extensions/transformer_engine.py`：

```python
if tp_comm_buffer_name == "qkv" and self.config.tp_comm_overlap_disable_qkv:
    extra_kwargs["ub_overlap_ag"] = False
    extra_kwargs["ub_overlap_rs_dgrad"] = False
```

### `tp_comm_overlap_disable_fc1`

禁用 MLP FC1 层上的 AllGather -> GEMM overlap。定义在 `third-party/megatron-lm/megatron/core/model_parallel_config.py`：

```python
tp_comm_overlap_disable_fc1: bool = False
"""
   If true, the AllGather -> Gemm overlap for FC1 layer of MLP gets disabled
"""
```

FC1 层通过 `tp_comm_buffer_name="fc1"` 标识，见 `third-party/megatron-lm/megatron/core/transformer/mlp.py`。当 buffer name 是 `fc1` 且该开关为真时，同样禁用 `ub_overlap_ag` 和 `ub_overlap_rs_dgrad`，见 `third-party/megatron-lm/megatron/core/extensions/transformer_engine.py`：

```python
if tp_comm_buffer_name == "fc1" and self.config.tp_comm_overlap_disable_fc1:
    extra_kwargs["ub_overlap_ag"] = False
    extra_kwargs["ub_overlap_rs_dgrad"] = False
```

### `--tp-comm-split-ag`

旧版 TE 兼容参数，TE v1.6.0 后已废弃。定义在 `third-party/megatron-lm/megatron/core/model_parallel_config.py`：

```python
tp_comm_split_ag: bool = True
"""Deprecated from TransformerEngine v1.6.0.
   Controls All-Gather overlap with Fprop GEMM by pipelining the GEMM and All-Gather
   splits. Don't care if tp_comm_overlap is False.
"""
```

在较旧 TE 路径中传给 `ub_split_ag`，见 `third-party/megatron-lm/megatron/core/extensions/transformer_engine.py`。

### `tp_comm_atomic_ag`

旧版 TE 兼容参数，TE v1.6.0 后已废弃。定义在 `third-party/megatron-lm/megatron/core/model_parallel_config.py`：

```python
tp_comm_atomic_ag: bool = False
"""Deprecated from TransformerEngine v1.6.0.
   If true, allows All-Gather overlap with Fprop GEMM by pipelining the GEMM and All-Gather
   both done atomically. Don't care if tp_comm_overlap is False.
"""
```

在较旧 TE 路径中传给 `ub_atomic_gemm_ag`，见 `third-party/megatron-lm/megatron/core/extensions/transformer_engine.py`。当前该字段在 CLI 自动生成列表中被排除，见 `third-party/megatron-lm/megatron/training/arguments.py`。

### `--tp-comm-split-rs`

旧版 TE 兼容参数，TE v1.6.0 后已废弃。定义在 `third-party/megatron-lm/megatron/core/model_parallel_config.py`：

```python
tp_comm_split_rs: bool = True
"""Deprecated from TransformerEngine v1.6.0.
   Controls Reduce-Scatter overlap with Fprop GEMM by pipelining the GEMM and
   Reduce-Scatter splits. Don't care if tp_comm_overlap is False.
"""
```

在较旧 TE 路径中传给 `ub_split_rs`，见 `third-party/megatron-lm/megatron/core/extensions/transformer_engine.py`。

### `tp_comm_atomic_rs`

旧版 TE 兼容参数，TE v1.6.0 后已废弃。定义在 `third-party/megatron-lm/megatron/core/model_parallel_config.py`：

```python
tp_comm_atomic_rs: bool = False
"""Deprecated from TransformerEngine v1.6.0.
   If true, allows Reduce-Scatter overlap with Fprop GEMM by pipelining the GEMM and
   Reduce-Scatter both done atomically. Don't care if tp_comm_overlap is False.
"""
```

在较旧 TE 路径中传给 `ub_atomic_gemm_rs`，见 `third-party/megatron-lm/megatron/core/extensions/transformer_engine.py`。当前该字段在 CLI 自动生成列表中被排除，见 `third-party/megatron-lm/megatron/training/arguments.py`。

### `--tp-comm-bootstrap-backend`

设置 TP overlap communicator 的 bootstrap backend。定义在 `third-party/megatron-lm/megatron/core/model_parallel_config.py`：

```python
tp_comm_bootstrap_backend: Literal['nccl', 'mpi', 'gloo'] = 'nccl'
"""Set the bootstrapping backend of Tensor parallel communications."""
```

初始化 UB 时传给 Transformer Engine，见 `third-party/megatron-lm/megatron/training/initialize.py`：

```python
te_module.base.initialize_ub(
    shape=input_shape,
    tp_size=args.tensor_model_parallel_size,
    quantization_modes=quantization_modes,
    ub_cfgs=ub_cfgs,
    bootstrap_backend=args.tp_comm_bootstrap_backend,
)
```

对较旧 TE，代码提示仅支持 MPI bootstrap backend，并创建 MPI process group，见 `third-party/megatron-lm/megatron/training/initialize.py`。

### `--decoder-tp-comm-overlap`

VLM 专用参数，只给 decoder / language transformer 开启 TP 通信与 GEMM overlap。CLI 定义在 `third-party/megatron-lm/pretrain_vlm.py`：

```python
group.add_argument("--decoder-tp-comm-overlap", action="store_true", default=False, help="Enables the overlap of "
                    "Tensor parallel communication and GEMM kernels in Decoder only. "
                    "Please provide decoder-seq-length when using this feature.")
```

`pretrain_vlm.py` 中会检查需要 Transformer Engine，并把该值写入 language transformer config，见 `third-party/megatron-lm/pretrain_vlm.py`：

```python
if args.decoder_tp_comm_overlap:
    assert args.transformer_impl == "transformer_engine", \
        "TransformerEngine is needed to support Decoder TP Comm overlap"
    language_transformer_config.tp_comm_overlap = args.decoder_tp_comm_overlap
```

同时 vision transformer 和 vision projection 中的 TP overlap 会被显式关闭，见 `third-party/megatron-lm/pretrain_vlm.py`。

### `--inference-fuse-tp-communication`

推理路径的 TP 通信融合开关，不是训练时的 UB overlap 参数，但与 TP 通信重叠/融合相关。定义在 `third-party/megatron-lm/megatron/core/transformer/transformer_config.py`：

```python
inference_fuse_tp_communication: bool = False
""" If true, uses a fused reduce-scatter-residual-norm-allgather kernel during inference. """
```

配置校验要求只能用于 `inference_optimized` transformer，且不支持 MoE，见 `third-party/megatron-lm/megatron/core/transformer/transformer_config.py`：

```python
if self.inference_fuse_tp_communication:
    assert self.transformer_impl == "inference_optimized"
    assert self.num_moe_experts is None
```

TransformerBlock 初始化时如果开启该开关，会调用 `_setup_fused_tp_communication()`，见 `third-party/megatron-lm/megatron/core/transformer/transformer_block.py`。该函数说明 fused 操作为 reduce-scatter + add + layer-norm + all-gather，见 `third-party/megatron-lm/megatron/core/transformer/transformer_block.py`。

## TP overlap 覆盖的层

TP overlap 依赖 `tp_comm_buffer_name` 标识不同线性层的 UB 名称。当前常见名称包括：

- attention output projection：`proj`，见 `third-party/megatron-lm/megatron/core/transformer/attention.py`
- attention QKV projection：`qkv`，见 `third-party/megatron-lm/megatron/core/transformer/attention.py`
- MLP FC1：`fc1`，见 `third-party/megatron-lm/megatron/core/transformer/mlp.py`
- MLP FC2：`fc2`，见 `third-party/megatron-lm/megatron/core/transformer/mlp.py`

TE Linear 对 UB 名称有白名单检查。若 `tp_comm_overlap` 开启，但 buffer name 不在 `qkv` / `proj` / `fc1` / `fc2` 中，会将该层持有配置的 `tp_comm_overlap` 设为 `False` 并给出 warning，见 `third-party/megatron-lm/megatron/core/extensions/transformer_engine.py`。

该机制只对 Transformer Engine 的 Linear / LayerNormLinear 生效，不作用于 Megatron local Linear。Megatron local `ColumnParallelLinear` / `RowParallelLinear` 虽然接口上保留了 `tp_comm_buffer_name`，但代码明确标注为 `Not used`，见 `third-party/megatron-lm/megatron/core/tensor_parallel/layers.py`。因此如果模型使用 `transformer_impl=local`，这些 `tp_comm_overlap*` 参数不会带来 TE UB 级别的 GEMM 与 TP 通信 overlap；Megatron local Linear 中只有基础的异步通信能力，例如异步 all-reduce 或 sequence-parallel 场景下的基础 reduce-scatter / all-gather 通信，不具备 TE UB 的 chunk pipeline / bulk overlap 调度。

## 配置示例

命令行最小示例：

```bash
--tensor-model-parallel-size 2 \
--transformer-impl transformer_engine \
--sequence-parallel \
--tp-comm-overlap \
--tp-comm-overlap-cfg /path/to/tp_overlap_ub.yaml \
--tp-comm-bootstrap-backend nccl
```

`--tp-comm-overlap-cfg` 指定的是 Transformer Engine UB 配置文件。仓库中的示例位于 `third-party/megatron-lm/tests/functional_tests/test_cases/gpt/gpt3_weekly_mcore_tp2_pp2_current_scaling_native_fp8_tp_pp_sp_tp_overlap/tp_comm_overlap_cfg.yaml`，以下摘录展示 pipeline 和 bulk 两种配置：

```yaml
proj_fprop:
  method: pipeline
  num_sm: 4
  num_splits: 4
  set_sm_margin: 0

fc1_dgrad:
  method: bulk
  num_sm: 2
  set_sm_margin: 0
```

## 使用注意事项

1. `tp_comm_overlap` 需要 `sequence_parallel=True`，否则启动阶段会断言失败。
2. 训练 TP overlap 依赖 `transformer_engine` 和 `yaml`，缺失时 `_initialize_tp_communicators()` 会抛出 RuntimeError。
3. `tp_comm_overlap_cfg` 只在 `tp_comm_overlap` 开启时有意义，用于初始化 Transformer Engine UB。
4. `tp_comm_bulk_*`、`tp_comm_overlap_*`、`tp_comm_split_*`、`tp_comm_atomic_*` 在 `tp_comm_overlap=False` 时都不生效。
5. `tp_comm_split_*` 和 `tp_comm_atomic_*` 是旧版 TE 兼容字段；新版本更应关注 `tp_comm_overlap_ag`、`tp_comm_overlap_rs`、`tp_comm_overlap_rs_dgrad`。
6. `decoder_tp_comm_overlap` 是 VLM decoder-only 开关，不等价于全模型 `tp_comm_overlap`。
7. `inference_fuse_tp_communication` 是推理 optimized transformer 的融合通信路径，不应和训练 TP UB overlap 混为同一类开关。
