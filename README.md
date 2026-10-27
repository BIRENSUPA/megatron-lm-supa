# 壁仞™ Megatron-LM-SUPA™

## 1. 简介

Megatron-LM-SUPA™ 是基于壁仞™开发的 Megatron-LM 适配仓库, 适配设备为BR2xx系列。通过安装本适配插件，用户可以在原生 Megatron-LM 代码仓库基础上实现对壁仞设备的支持。有关 Megatron-LM 的详细说明，请参考官方文档：[Megatron-LM GitHub](https://github.com/NVIDIA/Megatron-LM)

## 2. 环境搭建

### 2.1 拉取镜像

请先将 FULL_STACK_IMAGE 环境变量设置为可访问的壁仞完整软件栈镜像地址（含标签），然后拉取：

```bash
docker pull ${FULL_STACK_IMAGE}
```

### 2.2 启动容器

```bash
docker run -it --name [容器名称] --gpus all [镜像名称:标签] /bin/bash
```

## 3. 安装步骤

### 3.1 安装 torch_supa

```bash
pip install torch==2.12.0 torchvision==0.27.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cpu
pip install [torch_supa 包路径]
```

### 3.2 安装 flashattn_train

```bash
pip install [flashattn_train 包路径]
```

### 3.3 安装 megatron_supa

1. 拉取适配仓库（含 Megatron-LM 子模块）：

```bash
git clone --recursive https://github.com/BIRENSUPA/megatron-lm-supa.git 
cd megatron-lm-supa
export PYTHONPATH=${PWD}/third-party/megatron-lm:$PYTHONPATH
```

Megatron-LM 已作为 git submodule 包含在 `third-party/megatron-lm` 中，克隆时请加 `--recursive` 以自动拉取。

2. 安装 megatron_supa 适配插件：

```bash
pip install -r requirements.txt
pip install -e .
```

## 4. 已支持模型

详细规格见 [docs/supported_models.md](docs/supported_models.md)。

| 模型系列 | 支持规格 |
|----------|----------|
| GPT-3 | 175B |
| LLaMA 2 | 7B、13B |
| LLaMA 3 | 8B |
| LLaMA 3.1 | 8B、70B |
| Mixtral | 8x7B、8x22B |
| ChatGLM2 | 6B |
| Qwen1.5 | 0.5B、1.8B、4B、7B、14B、32B |
| Qwen2.5 | 72B |
| Qwen3 | 0.6B、1.7B、4B、8B、14B、32B |
| Qwen3.5 | Qwen3.5-35B-A3B |
| Qwen3.6 | Qwen3.6-27B |

## 5. 使用说明

### 5.1 导入 megatron_supa

在使用 Megatron-LM 前，需要先导入 megatron_supa 模块以启用壁仞设备支持：

```python
import megatron_supa
# 然后再导入 Megatron-LM 的其他模块
from megatron.training import get_args, pretrain
```

### 5.2 运行训练

使用与原生 Megatron-LM 相同的命令格式，例如：

```bash
python pretrain_gpt.py \
    --model-size-gb 1.5 \
    --num-layers 24 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --seq-length 1024 \
    --batch-size 8 \
    --train-iters 10000 \
    --save checkpoints \
    --load checkpoints \
    --data-path /path/to/data \
    --vocab-file /path/to/vocab.json \
    --merge-file /path/to/merges.txt \
    --distributed-backend gloo
```

## 6. 许可证与法律声明

- [LICENSE](LICENSE.md)：Apache License 2.0 许可证全文
- [版权声明](Copyright_Notice.md)
- [第三方开源软件声明](Third_Party_Open_Source_Software_Notice.md)

## 7. 贡献与安全

- [贡献指南](Contributing.zh.md)（[English](Contributing.md)），以及 [个人贡献者许可协议](<Individual Contributor License Agreement.md>)、[企业贡献者许可协议](<Corporate Contributor License Agreement.md>)
- [安全策略](SECURITY.zh.md)（[English](SECURITY.md)）
