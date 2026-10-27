# NCCL Allocator 作用、代码逻辑与 BR 适配说明

本文梳理 Megatron-LM 中 `nccl_allocator` 的作用方式、主要代码路径、训练场景、配置参数，以及在 BR 适配中的注意点。

## 1. NCCL allocator 的作用

`nccl_allocator` 用于为 DP/FSDP 通信 buffer 分配可注册到 NCCL 通信组的 user buffer。开启后，Megatron 会把部分长期存在的通信 buffer 放入由 NCCL allocator 创建的 memory pool，并将该 memory pool 注册给对应的通信组。

主要收益：

- 启用 NCCL user buffer registration，使 NCCL 可以对这些通信 buffer 使用更高效的通信路径。
- 降低部分 DP/FSDP 通信对 GPU SM 的占用，提升通信与计算 overlap 时的性能。
- 对 FSDP / Hybrid FSDP 中参数 all-gather、梯度 reduce-scatter、部分高精度梯度归约等场景更有效。
- 在 SHARP、跨 NVLink/IB 通信域、大模型 FSDP buffer 较大且反复通信的场景中收益更明显。

相关配置说明位于：

- `third-party/megatron-lm/megatron/core/distributed/distributed_data_parallel_config.py`

## 2. NCCL allocator 的代码逻辑

### 2.1 创建 NCCL allocator

Megatron 原生实现位置：

- `third-party/megatron-lm/megatron/core/nccl_allocator.py`

核心逻辑是通过 inline C++ extension 创建一个 `CUDAPluggableAllocator`：

- `third-party/megatron-lm/megatron/core/nccl_allocator.py`

原生 CUDA/NCCL 路径中，自定义 allocator 用 `ncclMemAlloc` 和 `ncclMemFree` 替换默认 malloc/free：

```cpp
void* nccl_alloc_plug(size_t size, int device, void* stream) {
    void* ptr;
    NCCL_CHECK(ncclMemAlloc(&ptr, size));
    return ptr;
}

void nccl_free_plug(void* ptr, size_t size, int device, void* stream) {
    NCCL_CHECK(ncclMemFree(ptr));
}

nccl_allocator = std::make_shared<
    torch::cuda::CUDAPluggableAllocator::CUDAPluggableAllocator>(
    nccl_alloc_plug, nccl_free_plug);
```

随后 Python 侧通过：

```python
nccl_allocator = torch.utils.cpp_extension.load_inline(...)
_allocator = nccl_allocator.get_nccl_allocator()
```

把 C++ allocator 保存到模块级 `_allocator`。

初始化入口：

- `third-party/megatron-lm/megatron/core/nccl_allocator.py`

```python
def init() -> None:
    os.environ["NCCL_NVLS_ENABLE"] = "1"
    os.environ["TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK"] = "0"
    _build_nccl_allocator()
```

其中：

- `NCCL_NVLS_ENABLE=1`：开启 NCCL NVLS 相关算法。
- `TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=0`：禁用 tensor register allocator hook，使用显式 memory pool 注册路径。

### 2.2 用 NCCL allocator 创建 memory pool

创建 memory pool 的入口：

- `third-party/megatron-lm/megatron/core/nccl_allocator.py`

```python
def create_nccl_mem_pool(symmetric=None):
    _build_nccl_allocator()
    assert _allocator is not None, "NCCL allocator is not initialized"
    if not symmetric:
        _pool = torch.cuda.MemPool(_allocator)
    else:
        _pool = torch.cuda.MemPool(_allocator, symmetric=symmetric)
    return _pool
```

实际代码会根据 PyTorch 版本兼容不同参数名：

- `symmetric`
- `symm_mem`
- 或者不在 `MemPool` 构造阶段传入 symmetric 参数，而在注册阶段传入。

使用方式通常是：

```python
nccl_allocator.init()
pool = nccl_allocator.create_nccl_mem_pool(
    symmetric=not ddp_config.disable_symmetric_registration
)
```

DP buffer 使用位置：

- `third-party/megatron-lm/megatron/core/distributed/param_and_grad_buffer.py`

FSDP buffer 使用位置：

- `third-party/megatron-lm/megatron/core/distributed/fsdp/src/megatron_fsdp/param_and_grad_buffer.py`

### 2.3 将 pool 中的 memory 注册给通信组

Megatron 提供了两类注册方式。

#### 方式一：上下文自动注册

单通信组注册类：

- `third-party/megatron-lm/megatron/core/nccl_allocator.py`

```python
with nccl_allocator.nccl_mem(pool, group=data_parallel_group, symmetric=True):
    buffer = torch.zeros(..., device="cuda")
```

关键逻辑：

- `__enter__` 阶段进入 `torch.cuda.use_mem_pool(self.pool)`，后续分配进入该 memory pool。
- 如果 pool 之前已有分配，会先 deregister，避免重复注册。
- `__exit__` 阶段调用通信 backend 的 `register_mem_pool` 把 pool 注册给通信组。

注册调用位置：

- `third-party/megatron-lm/megatron/core/nccl_allocator.py`

多通信组注册类：

- `third-party/megatron-lm/megatron/core/nccl_allocator.py`

```python
with nccl_allocator.MultiGroupMemPoolAllocator(pool, groups=[group_1, group_2]):
    buffer = torch.zeros(..., device="cuda")
```

用于 FSDP + EP、Hybrid FSDP 等同一 memory pool 需要注册到多个通信组的场景。

#### 方式二：手动注册

手动注册函数：

- `third-party/megatron-lm/megatron/core/nccl_allocator.py`

```python
def register_mem_pool(pool, group, symmetric=True):
    backend = group._get_backend(torch.device("cuda", torch.cuda.current_device()))
    if symmetric:
        try:
            backend.register_mem_pool(pool, symm=symmetric)
        except TypeError:
            backend.register_mem_pool(pool)
    else:
        backend.register_mem_pool(pool)
```

手动反注册函数：

- `third-party/megatron-lm/megatron/core/nccl_allocator.py`

FSDP 手动注册调用位置：

- `third-party/megatron-lm/megatron/core/distributed/fsdp/src/megatron_fsdp/param_and_grad_buffer.py`

训练循环中自动触发一次手动注册的位置：

- `third-party/megatron-lm/megatron/training/training.py`

```python
if (
    getattr(args, "fsdp_manual_registration", False)
    and getattr(args, "nccl_ub", False)
    and getattr(args, "use_megatron_fsdp", False)
    and iteration == start_iteration + 1
):
    for model_chunk in model:
        if isinstance(model_chunk, megatron_FSDP) and getattr(
            model_chunk.ddp_config, "fsdp_manual_registration", False
        ):
            param_and_grad_buffer = getattr(model_chunk, "param_and_grad_buffer", None)
            if param_and_grad_buffer is not None:
                param_and_grad_buffer.manual_buffer_registration()
```

## 3. Megatron 中的具体使用场景

### 3.1 普通 DP / Distributed Optimizer 通信 buffer

使用位置：

- `third-party/megatron-lm/megatron/core/distributed/distributed_data_parallel.py`
- `third-party/megatron-lm/megatron/core/distributed/param_and_grad_buffer.py`

当 `ddp_config.nccl_ub=True` 时，`_ParamAndGradBuffer` 会：

1. 调用 `nccl_allocator.init()`。
2. 创建 NCCL memory pool。
3. 构造 `nccl_allocator.nccl_mem(pool, group=data_parallel_group, ...)` 上下文。
4. 在上下文中分配 `param_data` 和 `grad_data`。

具体被放入 NCCL memory pool 的 buffer：

- `param_data`：当 `use_distributed_optimizer=True` 时创建，用于 distributed optimizer 的参数 all-gather / 参数同步相关路径。
- `grad_data`：梯度 contiguous buffer，用于 DP 梯度 reduce / reduce-scatter 相关路径。
- `shared_buffer`：MXFP8 特殊路径下，param AG 和 grad RS 共享 buffer。

相关代码：

- `third-party/megatron-lm/megatron/core/distributed/param_and_grad_buffer.py`

因此，在普通 DDP + distributed optimizer 中，`--use-nccl-ub` 主要影响参数/梯度通信 buffer 的分配和注册，使 DP 通信可使用 NCCL user buffer 路径。

### 3.2 Megatron FSDP 通信 buffer

使用位置：

- `third-party/megatron-lm/megatron/core/distributed/fsdp/src/megatron_fsdp/param_and_grad_buffer.py`

当 `nccl_ub=True` 时，FSDP 会：

1. 强制启用 `fsdp_double_buffer`，因为 user buffer registration 需要 persistent buffer。
2. 创建全局 `NCCL_MEMORY_POOL`。
3. 选择需要注册的通信组：
   - 普通 FSDP group。
   - Expert-DP group，存在 EP 时使用。
   - Outer/Inter-FSDP group，Hybrid FSDP 时使用。
   - independent all-gather group，开启独立 all-gather group 时使用。
4. 通过 `nccl_mem` 或 `MultiGroupMemPoolAllocator` 把 memory pool 注册给一个或多个通信组。

具体注册组选择位置：

- `third-party/megatron-lm/megatron/core/distributed/fsdp/src/megatron_fsdp/param_and_grad_buffer.py`

`fsdp_double_buffer` 默认使用 `FixedPoolAllocator(size=2)` 管理 FSDP 临时通信内存；开启 `megatron_fsdp_max_pool_double_buffer` 时改用 `MaxPoolAllocator`。每个 allocator 有两个 buffer group，用于在相邻或重叠的 FSDP unit 通信之间复用：

- `fsdp_params`：参数 all-gather 后的模型权重临时 buffer，对应 `model_weight_buffer`。
- `fsdp_fp8_transpose_params`：FP8/MXFP8 参数需要 transpose 时的权重临时 buffer，对应 `transpose_weight_buffer`。
- `fsdp_grads`：梯度 reduce-scatter / 梯度同步使用的主梯度通信 buffer，对应 `main_grad_buffer`。
- `hsdp_grad_comm`：Hybrid FSDP 中自定义梯度通信 dtype、高精度归约/低精度通信路径使用的额外梯度通信 buffer。

双缓冲池释放时只是把 buffer group 标记为空闲，底层 storage 会按名字保留在 global memory buffer 中继续复用和注册，并不会立即还给 CUDA/SUPA allocator。因此开启 `nccl_ub` 后即使当前只活跃一组通信，仍可能常驻两组参数/梯度通信临时 buffer；这也是 `--use-nccl-ub` 会额外增加显存占用的主要原因之一。

FSDP 中可能使用 NCCL user buffer 的 buffer 包括：

- `model_weight_buffer`：FSDP 参数 all-gather 使用的模型权重通信 buffer。
- `transpose_weight_buffer`：FP8/MXFP8 参数需要 transpose 时使用的权重通信 buffer。
- `main_weight_buffer`：distributed optimizer / sharded optimizer 使用的主权重 buffer。
- `main_grad_buffer`：梯度 reduce-scatter / 梯度同步使用的主梯度 buffer。
- `hsdp_grad_comm_alloc` 相关 buffer：Hybrid FSDP 中自定义梯度通信 dtype、高精度归约/低精度通信路径使用。

相关代码：

- `third-party/megatron-lm/megatron/core/distributed/fsdp/src/megatron_fsdp/param_and_grad_buffer.py`

对应通信用途：

- 参数 all-gather：把 sharded 参数聚合成当前计算需要的权重视图。
- 梯度 reduce-scatter：把 backward 产生的梯度规约并切分回各 rank。
- Hybrid FSDP outer/inner DP 同步：在 outer-DP 和 inner-DP layout 之间同步参数与梯度。
- EP 场景：expert 参数对应的 Expert-DP group 也需要注册。

## 4. 配置参数

### 4.1 `--use-nccl-ub`

定义位置：

- `third-party/megatron-lm/megatron/training/arguments.py`

作用：

- 开启 DP/FSDP 通信 buffer 的 NCCL user buffer registration。
- 普通 DP 中用于 `param_data`、`grad_data` 等 contiguous communication buffer。
- Megatron FSDP 中用于权重 all-gather、梯度 reduce-scatter、FSDP double buffer 等通信 buffer。

适用场景：

- 开启 distributed optimizer。
- 开启 Megatron FSDP。
- 通信与计算 overlap 明显，通信 kernel SM 占用影响训练性能。
- 大模型、多机、多通信域、SHARP 或 IB/NVL 混合通信环境。

注意：

- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 与 `nccl_ub` 不兼容，代码会直接报错。
- 相关检查位置：`third-party/megatron-lm/megatron/core/distributed/distributed_data_parallel_config.py`。
- Megatron FSDP 中开启 `nccl_ub` 会强制启用 `fsdp_double_buffer`，因为 NCCL user buffer registration 需要稳定、持久的通信 buffer 地址。

### 4.2 `--fsdp-double-buffer`

定义位置：

- `third-party/megatron-lm/megatron/training/arguments.py`
- `third-party/megatron-lm/megatron/core/distributed/distributed_data_parallel_config.py`

作用：

- 使用持久化双缓冲池保存 Megatron FSDP 通信中的临时参数和梯度 buffer。
- 每类临时通信 allocator 保留两组 buffer group，并在 FSDP unit 之间复用。
- 支持 NCCL user buffer registration 所需的稳定地址；启用 `nccl_ub` 时会自动强制开启。

保存内容：

- `fsdp_params`：参数 all-gather 后的模型权重临时 buffer。
- `fsdp_fp8_transpose_params`：FP8/MXFP8 transpose weight 临时 buffer。
- `fsdp_grads`：梯度 reduce-scatter / 同步使用的主梯度通信 buffer。
- `hsdp_grad_comm`：Hybrid FSDP 自定义梯度通信 dtype 路径的额外通信 buffer。

显存影响：

- 关闭时，FSDP 临时通信 buffer 可按需动态申请、resize 或释放。
- 开启时，双缓冲底层 storage 会按名字常驻缓存，只在逻辑上返回 idle pool。
- 因此显存更接近两组最大 FSDP unit 通信 buffer 的常驻开销，并且 FP8 transpose 或 HSDP 额外通信 buffer 会进一步增加占用。

限制：

- 当前主要适合 GPT 这类重复结构规整的模型。
- 不满足固定池复用条件的 FSDP unit 会退回动态 allocator，或在开启 fallback 时使用额外 persistent buffer。

### 4.3 `--disable-symmetric-registration`

定义位置：

- `third-party/megatron-lm/megatron/training/arguments.py`

作用：

- 禁用 symmetric/window registration。
- 强制使用 conventional/local user buffer registration。

适用场景：

- 当前 PyTorch/NCCL/后端不支持 symmetric registration。
- symmetric registration 出现兼容性问题、注册失败或性能不稳定。
- BR 后端尚未完整支持 symmetric 语义或需要临时规避。

影响：

```python
symmetric = not args.disable_symmetric_registration
```

该值会传给：

- `create_nccl_mem_pool(symmetric=...)`
- `nccl_mem(..., symmetric=...)`
- `MultiGroupMemPoolAllocator(..., symmetric=...)`
- `register_mem_pool(..., symmetric=...)`

### 4.4 `--fsdp-manual-registration`

定义位置：

- `third-party/megatron-lm/megatron/training/arguments.py`

作用：

- FSDP 场景下手动注册通信 buffer 到 NCCL user buffer。
- 只在 `--use-megatron-fsdp` 且 `--use-nccl-ub` 时有效。

适用场景：

- 大模型 FSDP 中 symmetric registration 本身耗时明显。
- 希望减少反复 register/deregister 的次数。
- FSDP double buffer 已经稳定创建，适合在 warmup 后一次性注册。

代码路径：

- `get_mem_alloc_context()` 中，如果 `fsdp_manual_registration=True`，返回 `MemPoolAllocatorWithoutRegistration`，即只使用 memory pool 分配，不在上下文退出时自动注册。
- 训练循环在 `start_iteration + 1` 后调用 `manual_buffer_registration()` 完成手动注册。

相关位置：

- `third-party/megatron-lm/megatron/core/distributed/fsdp/src/megatron_fsdp/param_and_grad_buffer.py`
- `third-party/megatron-lm/megatron/core/nccl_allocator.py`
- `third-party/megatron-lm/megatron/training/training.py`

## 5. BR 适配注意点

### 5.1 BR 会 patch Megatron 原生 `_build_nccl_allocator`

BR adaptor patch 位置：

- `megatron_supa/patchs/megatron_patch.py`

```python
def patch_nccl_allocator():
    from ..core.nccl_allocator import _build_nccl_allocator
    PatchManager.register_patch(
        'megatron.core.nccl_allocator._build_nccl_allocator',
        _build_nccl_allocator,
    )
```

因此，业务代码仍然 import 和调用：

```python
import megatron.core.nccl_allocator as nccl_allocator
```

但实际 `_build_nccl_allocator()` 会被替换成 BR 版本。

前提是必须先导入 `megatron_supa`：

```python
import megatron_supa
from megatron.training import pretrain
```

如果没有先导入 `megatron_supa`，Megatron 会走原生 CUDA/NCCL allocator 路径，不会使用 BR 的 BCCL allocator。

### 5.2 BR 版本 allocator 使用 BCCL 和 torch_supa SUPA allocator

BR 实现位置：

- `megatron_supa/core/nccl_allocator.py`

BR 版本不再使用：

```cpp
ncclMemAlloc
ncclMemFree
CUDAPluggableAllocator
```

而是替换为：

```cpp
bcclMemAlloc
bcclMemFree
torch_supa::supa::SUPAPluggableAllocator
```

相关代码位置：

- `megatron_supa/core/nccl_allocator.py`

BR inline extension 链接参数：

- `megatron_supa/core/nccl_allocator.py`

```python
torch_supa_lib_path = os.path.join(PYTORCH_SUPA_INSTALL_PATH, "lib")
extra_ldflags=["-lbccl", f"-L{torch_supa_lib_path} -ltorch_supa"]
```

### 5.3 BR 使用时的环境要求

BR allocator 依赖：

- `torch_supa` 可 import。
- BCCL 库可链接。
- `torch_supa` 安装路径下存在对应头文件和动态库；`PYTORCH_SUPA_INSTALL_PATH` 由 `torch_supa.__file__` 自动推导。
- 编译 inline extension 的环境完整。

如果 `torch_supa` 不存在或路径不正确，BR allocator 无法构建。编译目录也需要可写：优先使用 `TORCH_EXTENSIONS_DIR`，否则使用 `~/.cache/torch_extensions/nccl_allocator`；上述缓存不可用时才回退到包内的 `build/` 目录。

### 5.4 BR 适配中的语义保持

BR patch 只替换 allocator 的底层构建方式，上层 Megatron 使用语义保持不变：

- `nccl_allocator.init()`
- `nccl_allocator.create_nccl_mem_pool()`
- `nccl_allocator.nccl_mem(...)`
- `nccl_allocator.MultiGroupMemPoolAllocator(...)`
- `nccl_allocator.register_mem_pool(...)`

也就是说，Megatron 的 DP/FSDP 代码仍按 NCCL allocator API 编写；BR 只把底层 allocation/free 从 CUDA/NCCL 替换为 SUPA/BCCL。

### 5.5 BR 适配排查建议

排查 `--use-nccl-ub` 相关问题时优先确认：

1. 训练入口是否先 `import megatron_supa`。
2. patch 是否生效：`megatron.core.nccl_allocator._build_nccl_allocator` 是否来自 `megatron_supa.core.nccl_allocator`。
3. `torch_supa` 是否可 import，其自动推导的安装目录中是否有对应头文件和动态库。
4. inline extension 是否成功链接 `bccl` 和 `torch_supa`。
5. 如 symmetric registration 有兼容问题，先加 `--disable-symmetric-registration` 验证 local registration 路径。
6. FSDP 大模型注册耗时明显时，再评估 `--fsdp-manual-registration`。

## 6. 最小配置示例

普通 DP / distributed optimizer：

```bash
--use-distributed-optimizer \
--use-nccl-ub
```

规避 symmetric registration：

```bash
--use-distributed-optimizer \
--use-nccl-ub \
--disable-symmetric-registration
```

Megatron FSDP + 手动注册：

```bash
--use-megatron-fsdp \
--use-nccl-ub \
--fsdp-manual-registration
```

BR 环境下务必保证训练入口先导入：

```python
import megatron_supa
```
