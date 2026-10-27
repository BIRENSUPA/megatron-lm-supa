#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${SCRIPT_DIR}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third-party/megatron-lm:${PYTHONPATH}"


GPUS_PER_NODE=2
# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6001"}
NNODES=${SLURM_NNODES:-"1"}
NODE_RANK=${RANK:-"0"}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

# DATA_PATH is the Megatron mmap dataset prefix (without the .bin/.idx suffix), e.g.:
#   DATA_PATH=/your/datasets/RedPajama-Data-1T-Sample/redpajama-llama2_text_document
DATA_PATH=${DATA_PATH:-<PATH_TO_DATASETS>/RedPajama-Data-1T-Sample/redpajama-llama2_text_document}
DATA_CACHE_PATH=${DATA_CACHE_PATH:-<PATH_TO_DATASETS>/RedPajama-Data-1T-Sample/redpajama-llama2_text_document/index-cache}

TP_SIZE=2
PP_SIZE=1
GLOBAL_BATCH=16
MICRO_BATCH=1
SEQ_LEN=128
MODEL_SIZE=lite

# Model config for lite size
NUM_LAYERS=1
HIDDEN_SIZE=128
NUM_ATTN_HEADS=2
INTERMEDIATE_SIZE=11008
MAX_POSITION_EMBEDDINGS=32768
PADDED_VOCAB_SIZE=151936
ROTARY_BASE=1000000
GQA_OPTIONS=""

# 设置CUDA可见设备
export CUDA_VISIBLE_DEVICES=1,2

# 加载环境变量
source "${SCRIPT_DIR}/env.sh"

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

MODEL_ARGS=(
    --disable-bias-linear
    --seq-length ${SEQ_LEN}
    --max-position-embeddings ${MAX_POSITION_EMBEDDINGS}
    --num-layers ${NUM_LAYERS}
    --hidden-size ${HIDDEN_SIZE}
    --ffn-hidden-size ${INTERMEDIATE_SIZE}
    --num-attention-heads ${NUM_ATTN_HEADS}
    --init-method-std 0.008
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --norm-epsilon 1e-06
    --position-embedding-type rope
    --swiglu
    --untie-embeddings-and-output-weights
    --use-rotary-position-embeddings
    --no-rope-fusion
    --rotary-percent 1.0
    --rotary-base ${ROTARY_BASE}
    --rotary-seq-len-interpolation-factor 1
    # --add-qkv-bias
    --transformer-impl transformer_engine
    --distributed-timeout-minutes 1000
    # --attention-backend fused
    --use-distributed-optimizer
    --no-gradient-accumulation-fusion
    --no-bias-dropout-fusion
    --exit-interval 5 
    --distributed-timeout-minutes 1000
)

DATA_ARGS=(
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model "${TOKENIZER_MODEL}"
    --split 99,1,0
    --data-path ${DATA_PATH}
    --data-cache-path ${DATA_CACHE_PATH}
    --padded-vocab-size ${PADDED_VOCAB_SIZE}
)

TRAINING_ARGS=(
    --micro-batch-size ${MICRO_BATCH}
    --global-batch-size ${GLOBAL_BATCH}
    --lr 1e-5
    --train-iters 5000000 
    --lr-decay-iters 5000000 
    --lr-decay-style cosine
    --min-lr 1e-6
    --weight-decay 0.1
    --lr-warmup-iters 0 
    --clip-grad 1.0
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --bf16
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size ${TP_SIZE}
    --pipeline-model-parallel-size ${PP_SIZE}
    --sequence-parallel
)

LOGGING_ARGS=(
    --log-interval 1
    --save-interval 100000
    --eval-interval 10000
    --eval-iters 10
    --save $CHECKPOINT_PATH
    --tensorboard-dir "${TENSORBOARD_PATH}"
    --no-load-optim
    --no-load-rng
)

 torchrun ${DISTRIBUTED_ARGS[@]} model/pretrain_qwen.py \
    ${MODEL_ARGS[@]} \
    ${GQA_OPTIONS} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]} \
    2>&1 | tee -a ${LOGS_PATH}/${timestamp}.log
