# 已支持模型列表

本文档列出 Megatron-LM-BR2 适配仓库当前支持的模型系列及其规格，信息来源于 `examples/` 目录下的训练脚本。

## 模型汇总

| 模型系列 | 支持规格 | 架构类型 | 精度 | 训练示例路径 |
|----------|----------|----------|------|--------------|
| GPT-3 | 175B | Dense | BF16 | [examples/gpt3/](../examples/gpt3/) |
| LLaMA 2 | 7B、13B | Dense | BF16 | [examples/llama2/](../examples/llama2/) |
| LLaMA 3 | 8B | Dense | BF16 | [examples/llama3/](../examples/llama3/) |
| LLaMA 3.1 | 8B、70B | Dense | BF16 | [examples/llama3/llama3_1/](../examples/llama3/llama3_1/) |
| Mixtral | 8x7B、8x22B | MoE | BF16 | [examples/mixtral/](../examples/mixtral/) |
| ChatGLM2 | 6B | Dense | BF16 | [examples/chatglm2/](../examples/chatglm2/) |
| Qwen1.5 | 0.5B、1.8B、4B、7B、14B、32B | Dense | BF16 | [examples/qwen1_5/](../examples/qwen1_5/) |
| Qwen3 | 0.6B、1.7B、4B、8B、14B、32B | Dense | BF16 | [examples/qwen3/](../examples/qwen3/) |
| Qwen3-MoE | MoE-~30B | MoE | BF16 | [examples/qwen3/](../examples/qwen3/) |

---

## 详细规格

### GPT-3

训练脚本：[examples/gpt3/pretrain_gpt3_175b_MB1_GB1536_TP8_PP16_DP1.sh](../examples/gpt3/pretrain_gpt3_175b_MB1_GB1536_TP8_PP16_DP1.sh)

| 规格 | 层数 | 隐藏维度 | 注意力头 | FFN 维度 | 词表大小 | 最大序列长 |
|------|------|----------|----------|----------|----------|----------|
| 175B | 96   | 12288    | 96       | 49152    | 50257    | 2048     |

- Tokenizer：`GPT2BPETokenizer`
- 位置编码：Learned Absolute
- 归一化：LayerNorm
- 激活函数：GeLU

---

### LLaMA 2

训练脚本：[examples/llama2/pretrain_llama2_MB1_GB2_TP2_PP1_DP1_SP_N1C2.sh](../examples/llama2/pretrain_llama2_MB1_GB2_TP2_PP1_DP1_SP_N1C2.sh)

| 规格 | 层数 | 隐藏维度 | 注意力头 | FFN 维度 | 词表大小 | 最大序列长 |
|------|------|----------|----------|----------|----------|----------|
| 7B   | 32   | 4096     | 32       | 11008    | 32000    | 4096     |
| 13B  | 40   | 5120     | 40       | 13824    | 32000    | 4096     |

- Tokenizer：`Llama2Tokenizer`
- 位置编码：RoPE（base=10000）
- 归一化：RMSNorm
- 激活函数：SwiGLU

---

### LLaMA 3

训练脚本：[examples/llama3/pretrain_llama3_8b_MB1_GB2_L1_TP2_PP1_DP1_SP_N1C2.sh](../examples/llama3/pretrain_llama3_8b_MB1_GB2_L1_TP2_PP1_DP1_SP_N1C2.sh)

| 规格 | 层数 | 隐藏维度 | 注意力头 | KV 头 | FFN 维度 | 词表大小 | 最大序列长 |
|------|------|----------|----------|-------|----------|----------|----------|
| 8B   | 32   | 4096     | 32       | 8     | 14336    | 128256   | 8192     |

- Tokenizer：`HuggingFaceTokenizer`
- 位置编码：RoPE（base=500000）
- GQA：是
- 归一化：RMSNorm
- 激活函数：SwiGLU

---

### LLaMA 3.1

训练脚本：[examples/llama3/llama3_1/pretrain_llama3_1_8b_MB1_GB2_L1_TP2_PP1_DP1_SP_N1C2.sh](../examples/llama3/llama3_1/pretrain_llama3_1_8b_MB1_GB2_L1_TP2_PP1_DP1_SP_N1C2.sh)

| 规格 | 层数 | 隐藏维度 | 注意力头 | KV 头 | FFN 维度 | 词表大小 | 最大序列长 |
|------|------|----------|----------|-------|----------|----------|----------|
| 8B   | 32   | 4096     | 32       | 8     | 14336    | 128256   | 131072   |
| 70B  | 80   | 8192     | 64       | 8     | 28672    | 128256   | 131072   |

- Tokenizer：`HuggingFaceTokenizer`
- 位置编码：RoPE（base=500000，scaling_factor=8）
- GQA：是
- 归一化：RMSNorm
- 激活函数：SwiGLU

---

### Mixtral（MoE）

训练脚本：[examples/mixtral/pretrain_mixtral_MB1_GB2_L1_TP1_PP1_DP1_EP2_SP_N1C2.sh](../examples/mixtral/pretrain_mixtral_MB1_GB2_L1_TP1_PP1_DP1_EP2_SP_N1C2.sh)

| 规格  | 层数 | 隐藏维度 | 注意力头 | KV 头 | FFN 维度 | 专家数 | Top-K | 最大序列长 |
|-------|------|----------|----------|-------|----------|--------|-------|----------|
| 8x7B  | 32   | 4096     | 32       | 8     | 14336    | 8      | 2     | 32768    |
| 8x22B | 56   | 6144     | 48       | 8     | 16384    | 8      | 2     | 65536    |

- Tokenizer：`Llama2Tokenizer`
- 位置编码：RoPE（base=1000000）
- GQA：是
- MoE 路由：aux_loss 负载均衡
- 归一化：RMSNorm

---

### ChatGLM2

训练脚本：[examples/chatglm2/pretrain_chatglm2_6b_MB1_GB2_L1_TP2_PP1_DP1_SP_N1C2.sh](../examples/chatglm2/pretrain_chatglm2_6b_MB1_GB2_L1_TP2_PP1_DP1_SP_N1C2.sh)

| 规格 | 层数 | 隐藏维度 | 注意力头 | KV 头 | FFN 维度 | 词表大小 | 最大序列长 |
|------|------|----------|----------|-------|----------|----------|----------|
| 6B   | 28   | 4096     | 32       | 2     | 13696    | 65024    | 32768    |

- Tokenizer：`HuggingFaceTokenizer`
- 位置编码：RoPE（original_rope）
- GQA（MQA）：是
- 归一化：RMSNorm

---

### Qwen1.5

训练脚本：[examples/qwen1_5/pretrain.sh](../examples/qwen1_5/pretrain.sh)

| 规格  | 层数 | 隐藏维度 | 注意力头 | FFN 维度 | 词表大小 | 最大序列长 |
|-------|------|----------|----------|----------|----------|----------|
| 0.5B  | 24   | 1024     | 16       | 2816     | 151936   | 32768    |
| 1.8B  | 24   | 2048     | 16       | 5504     | 151936   | 32768    |
| 4B    | 40   | 2560     | 20       | 6912     | 151936   | 32768    |
| 7B    | 32   | 4096     | 32       | 11008    | 151936   | 32768    |
| 14B   | 40   | 5120     | 40       | 13696    | 151936   | 32768    |
| 32B   | 64   | 5120     | 40       | 27392    | 151936   | 32768    |

- Tokenizer：`HuggingFaceTokenizer`
- 位置编码：RoPE（base=1000000）
- 32B 采用 GQA（8 KV heads）；其余规格不使用 GQA
- 归一化：RMSNorm，norm-epsilon=1e-6
- 激活函数：SwiGLU，含 QKV Bias

---

### Qwen3（Dense）

训练脚本：[examples/qwen3/pretrain_qwen3_MB1_GB2_L1_TP2_PP1_DP1_SP_N1C2.sh](../examples/qwen3/pretrain_qwen3_MB1_GB2_L1_TP2_PP1_DP1_SP_N1C2.sh)

| 规格  | 层数 | 隐藏维度 | 注意力头 | KV 头 | FFN 维度 | 词表大小 | 最大序列长 |
|-------|------|----------|----------|-------|----------|----------|----------|
| 0.6B  | 28   | 1024     | 16       | 8     | 3072     | 151936   | 40960    |
| 1.7B  | 28   | 2048     | 16       | 8     | 6144     | 151936   | 40960    |
| 4B    | 36   | 2560     | 32       | 8     | 9728     | 151936   | 40960    |
| 8B    | 36   | 4096     | 32       | 8     | 12288    | 151936   | 40960    |
| 14B   | 40   | 5120     | 40       | 8     | 17408    | 151936   | 40960    |
| 32B   | 64   | 5120     | 64       | 8     | 25600    | 151936   | 40960    |

- Tokenizer：`HuggingFaceTokenizer`
- 位置编码：RoPE（base=1000000）
- GQA：是
- 归一化：RMSNorm，含 QK LayerNorm
- 激活函数：SwiGLU

---

### Qwen3-MoE

训练脚本：[examples/qwen3/pretrain_qwen3_moe_32b_MB1_GB2_L1_TP2_PP1_DP1_SP_N1C2.sh](../examples/qwen3/pretrain_qwen3_moe_32b_MB1_GB2_L1_TP2_PP1_DP1_SP_N1C2.sh)

| 规格     | 层数 | 隐藏维度 | 注意力头 | KV 头 | 专家数 | Top-K | MoE FFN 维度 | 最大序列长 |
|----------|------|----------|----------|-------|--------|-------|--------------|----------|
| MoE-~30B | 48   | 2048     | 32       | 4     | 128    | 8     | 768          | 40960    |

- Tokenizer：`HuggingFaceTokenizer`
- 位置编码：RoPE（base=1000000）
- GQA：是（4 KV heads）
- 归一化：RMSNorm，含 QK LayerNorm
- MoE 路由：aux_loss 负载均衡，含 grouped-gemm 优化
