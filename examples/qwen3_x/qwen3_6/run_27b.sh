#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QWEN3_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${QWEN3_ROOT}"

export NCCL_NVLS_ENABLE=0
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1

# DATA_PATH is the Megatron mmap dataset prefix (without the .bin/.idx suffix), e.g.:
#   DATA_PATH=/your/datasets/RedPajama-Data-1T-Sample/redpajama-llama2_text_document
bash "${QWEN3_ROOT}/train_qwen.sh" \
  --model_size 27B --tp 8 --pp 1 --cp 1 --nproc_per_node 8 --node_num 1 --node_rank 0 \
  --mbs 1 --mbn 128 --seq_len 32768 --train_iters 5 \
  --data_source "${DATA_PATH:?Set DATA_PATH to the Megatron dataset prefix}" \
  --tokenizer_path "${TOKENIZER_PATH:?Set TOKENIZER_PATH to the matching model tokenizer directory}"
