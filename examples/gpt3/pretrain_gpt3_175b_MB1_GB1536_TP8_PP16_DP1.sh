#!/bin/bash

# Pretrain GPT-3 175B on Biren GPUs.
# Parallelism: TP=8, PP=16, DP=1, SP enabled
# Reference: https://arxiv.org/abs/2005.14165
# - Layers: 96, Hidden: 12288, Heads: 96, Seq: 2048
# - Global batch: 1536, micro batch: 1
# Usage:
#   DATA_PATH=<prefix>_text_document bash pretrain_gpt3_175b_MB1_GB1536_TP8_PP16_DP1.sh


SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${SCRIPT_DIR}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third-party/megatron-lm:${PYTHONPATH}"

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6006"}
NNODES=${SLURM_NNODES:-"1"}
NODE_RANK=${RANK:-"0"}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

# DATA_PATH is the Megatron mmap dataset prefix (without the .bin/.idx suffix), e.g.:
#   DATA_PATH=/your/datasets/RedPajama-Data-1T-Sample/redpajama-llama2_text_document
DATA_PATH=${DATA_PATH:-<PATH_TO_DATASETS>/RedPajama-Data-1T-Sample/redpajama-llama2_text_document}
DATA_CACHE_PATH=${DATA_CACHE_PATH:-<PATH_TO_DATASETS>/RedPajama-Data-1T-Sample/redpajama-llama2_text_document/index-cache}

TP_SIZE=8
PP_SIZE=16
GLOBAL_BATCH=1536
MICRO_BATCH=1
SEQ_LEN=2048
MODEL_SIZE=175B

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# GPT-3 175B architecture
# Reference: https://arxiv.org/abs/2005.14165 Table 2
NLAYERS=96
HIDDEN_SIZE=12288
NUM_ATTENTION_HEADS=96
VOCAB_SIZE=50257

source "${SCRIPT_DIR}/env.sh"

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

MODEL_ARGS=(
    --num-layers ${NLAYERS}
    --hidden-size ${HIDDEN_SIZE}
    --num-attention-heads ${NUM_ATTENTION_HEADS}
    --seq-length ${SEQ_LEN}
    --max-position-embeddings ${SEQ_LEN}
    --init-method-std 0.006
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization LayerNorm
    --position-embedding-type learned_absolute
    --make-vocab-size-divisible-by 128
    --transformer-impl transformer_engine
    --no-gradient-accumulation-fusion
    --no-bias-dropout-fusion
    --attention-backend fused
    --exit-interval 5
    --distributed-timeout-minutes 1000
)

DATA_ARGS=(
    --vocab-file $VOCAB_FILE
    --merge-file $MERGE_FILE
    --data-path $DATA_PATH
    --data-cache-path $DATA_CACHE_PATH
    --split 949,50,1
)

TRAINING_ARGS=(
    --micro-batch-size ${MICRO_BATCH}
    --global-batch-size ${GLOBAL_BATCH}
    --rampup-batch-size 16 16 5859375
    --lr 6.0e-5
    --train-iters 500000
    --lr-decay-iters 430000
    --lr-decay-style cosine
    --min-lr 6.0e-6
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --lr-warmup-fraction 0.001
    --clip-grad 1.0
    --bf16
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size ${TP_SIZE}
    --pipeline-model-parallel-size ${PP_SIZE}
    --sequence-parallel
)

LOGGING_ARGS=(
    --log-interval 100
    --save-interval 10000
    --eval-interval 1000
    --eval-iters 10
    --tensorboard-dir "${TENSORBOARD_PATH}"
    --no-load-optim
    --no-load-rng
)

if [ -n "${WANDB_API_KEY}" ]; then
    LOGGING_ARGS+=(
        --wandb-project ${WANDB_PROJECT:-"GPT3"}
        --wandb-exp-name ${WANDB_NAME:-"GPT3_175B"}
    )
fi

torchrun ${DISTRIBUTED_ARGS[@]} model/pretrain_gpt3.py \
    ${MODEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]} \
    2>&1 | tee -a ${LOGS_PATH}/${timestamp}.log
