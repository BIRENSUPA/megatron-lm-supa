# Biren™ Megatron-LM-SUPA™

[中文](README.md) | English

## 1. Introduction

Megatron-LM-SUPA™ is a Megatron-LM adaptation repository developed by Biren™, targeting the BR2xx series devices. By installing this adaptation plugin, users can enable Biren device support on top of the native Megatron-LM codebase. For details about Megatron-LM, please refer to the official documentation: [Megatron-LM GitHub](https://github.com/NVIDIA/Megatron-LM)

## 2. Environment Setup

### 2.1 Pull the Image

Set the FULL_STACK_IMAGE environment variable to an accessible Biren full-stack image address (including the tag), then pull it:

```bash
docker pull ${FULL_STACK_IMAGE}
```

### 2.2 Start the Container

```bash
docker run -it --name [container_name] --gpus all [image_name:tag] /bin/bash
```

## 3. Installation

### 3.1 Install torch_supa

```bash
pip install torch==2.12.0 torchvision==0.27.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cpu
pip install [path_to_torch_supa_package]
```

### 3.2 Install flashattn_train

```bash
pip install [path_to_flashattn_train_package]
```

### 3.3 Install megatron_supa

1. Clone the adaptation repository (including the Megatron-LM submodule):

```bash
git clone --recursive https://github.com/BIRENSUPA/megatron-lm-supa.git 
cd megatron-lm-supa
export PYTHONPATH=${PWD}/third-party/megatron-lm:$PYTHONPATH
```

Megatron-LM is included as a git submodule under `third-party/megatron-lm`. Use `--recursive` when cloning so that it is fetched automatically.

2. Install the megatron_supa adaptation plugin:

```bash
pip install -r requirements.txt
pip install -e .
```

## 4. Supported Models

See [docs/supported_models.md](docs/supported_models.md) for detailed specifications.

| Model Family | Supported Sizes |
|--------------|-----------------|
| GPT-3 | 175B |
| LLaMA 2 | 7B, 13B |
| LLaMA 3 | 8B |
| LLaMA 3.1 | 8B, 70B |
| Mixtral | 8x7B, 8x22B |
| ChatGLM2 | 6B |
| Qwen1.5 | 0.5B, 1.8B, 4B, 7B, 14B, 32B |
| Qwen2.5 | 72B |
| Qwen3 | 0.6B, 1.7B, 4B, 8B, 14B, 32B |
| Qwen3.5 | Qwen3.5-35B-A3B |
| Qwen3.6 | Qwen3.6-27B |

## 5. Usage

### 5.1 Import megatron_supa

Before using Megatron-LM, import the megatron_supa module first to enable Biren device support:

```python
import megatron_supa
# Then import other Megatron-LM modules
from megatron.training import get_args, pretrain
```

### 5.2 Run Training

Use the same command format as native Megatron-LM, for example:

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

## 6. License and Legal Notices

- [LICENSE](LICENSE.md): Full text of the Apache License 2.0
- [Copyright Notice](Copyright_Notice.md)
- [Third-Party Open Source Software Notice](Third_Party_Open_Source_Software_Notice.md)

## 7. Contributing and Security

- [Contributing Guide](Contributing.md) ([中文](Contributing.zh.md)), along with the [Individual Contributor License Agreement](<Individual Contributor License Agreement.md>) and [Corporate Contributor License Agreement](<Corporate Contributor License Agreement.md>)
- [Security Policy](SECURITY.md) ([中文](SECURITY.zh.md))
