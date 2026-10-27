#!/bin/bash


SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
cd "${SCRIPT_DIR}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third-party/megatron-lm:${PYTHONPATH}"

export NVTE_FUSED_ATTN=1
export NVTE_FLASH_ATTN=0

# export NVTE_DEBUG=1 
# export NVTE_DEBUG_LEVEL=1

GPUS_PER_NODE=8
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6002"}
NNODES=${SLURM_NNODES:-"1"}
NODE_RANK=${RANK:-"0"}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

# DATA_PATH is the Megatron mmap dataset prefix (without the .bin/.idx suffix), e.g.:
#   DATA_PATH=/your/datasets/RedPajama-Data-1T-Sample/redpajama-llama2_text_document
DATA_PATH=${DATA_PATH:-<PATH_TO_DATASETS>/RedPajama-Data-1T-Sample/redpajama-llama2_text_document}
DATA_CACHE_PATH=${DATA_CACHE_PATH:-<PATH_TO_DATASETS>/RedPajama-Data-1T-Sample/redpajama-llama2_text_document/index-cache}

TP_SIZE=8
PP_SIZE=1
GLOBAL_BATCH=2
MICRO_BATCH=2
SEQ_LEN=4096
MODEL_SIZE=70B_tp8

LLAMA3_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

NLAYERS=1
HIDDEN_SIZE=8192
NUM_ATTENTION_HEADS=64
INTERMEDIATE_SIZE=28672
NUM_KEY_VALUE_HEADS=8
VOCAB_SIZE=128256
MAX_POSITION_EMBEDDINGS=131072
GQA_OPTIONS=" \
            --group-query-attention \
            --num-query-groups ${NUM_KEY_VALUE_HEADS}"

source "${SCRIPT_DIR}/env.sh"

MODEL_ARGS=(
    --disable-bias-linear
    --seq-length ${SEQ_LEN}
    --max-position-embeddings ${MAX_POSITION_EMBEDDINGS}
    --num-layers ${NLAYERS}
    --hidden-size ${HIDDEN_SIZE}
    --ffn-hidden-size ${INTERMEDIATE_SIZE}
    --num-attention-heads ${NUM_ATTENTION_HEADS}
    --init-method-std 0.008
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --position-embedding-type rope
    --swiglu
    --untie-embeddings-and-output-weights
    --no-position-embedding
    --rotary-base 500000
    --use-rope-scaling
    --rope-scaling-factor 8
    --padded-vocab-size ${VOCAB_SIZE}
    --transformer-impl transformer_engine
    --no-gradient-accumulation-fusion
    --no-bias-dropout-fusion
    --exit-interval 5
    --distributed-timeout-minutes 1000
    --attention-backend fused 
    --profile
    --use-pytorch-profiler 
    --pytorch-profiler-collect-shapes 
    --pytorch-profiler-collect-callstack 
    --profile-step-start 1
    --profile-step-end 2
)

DATA_ARGS=(
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model "${TOKENIZER_MODEL}"
    --data-path $DATA_PATH
    --data-cache-path $DATA_CACHE_PATH
    --split 99990,8,2
)

TRAINING_ARGS=(
    --micro-batch-size ${MICRO_BATCH}
    --global-batch-size ${GLOBAL_BATCH}
    --lr 1.5e-4
    --train-iters 500000
    --lr-decay-iters 320000
    --lr-decay-style cosine
    --min-lr 1.5e-5
    --weight-decay 0.1
    --lr-warmup-iters 500
    --clip-grad 1.0
    --bf16
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size ${TP_SIZE}
    --pipeline-model-parallel-size ${PP_SIZE}
    --expert-model-parallel-size 1
    --sequence-parallel
)

LOGGING_ARGS=(
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10 \
    --tensorboard-dir "${TENSORBOARD_PATH}" \
    --no-load-optim \
    --no-load-rng
)

if [ -n "${WANDB_API_KEY}" ]; then
    LOGGING_ARGS+=(
        --wandb-project ${WANDB_PROJECT:-"Llama3_1"}
        --wandb-exp-name ${WANDB_NAME:-"Llama3_1_${MODEL_SIZE}"}
    )
fi

torchrun ${DISTRIBUTED_ARGS[@]} ${LLAMA3_DIR}/model/pretrain_llama.py \
    ${MODEL_ARGS[@]} \
    ${GQA_OPTIONS} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]} \
    2>&1 | tee -a ${LOGS_PATH}/${timestamp}.log
