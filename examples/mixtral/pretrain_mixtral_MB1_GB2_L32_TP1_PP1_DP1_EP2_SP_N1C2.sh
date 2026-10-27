#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${SCRIPT_DIR}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third-party/megatron-lm:${PYTHONPATH}"


# Runs Mixtral model
# Supports MODEL_SIZE in: 8x7b, 8x22b
# To specify model size, run:
# MODEL_SIZE=8x7b bash pretrain_mixtral_MB1_GB2_L1_TP1_PP1_DP1_EP2_SP_N1C2.sh
# MODEL_SIZE=8x22b bash pretrain_mixtral_MB1_GB2_L1_TP1_PP1_DP1_EP2_SP_N1C2.sh
# 8x7b config source:
# https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/blob/main/config.json
# 8x22b config source:
# https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1/blob/main/config.json

# export CUDA_VISIBLE_DEVICES=4,5,6,7

GPUS_PER_NODE=2
# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6000"}
NNODES=${SLURM_NNODES:-"1"}
NODE_RANK=${RANK:-"0"}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

# DATA_PATH is the Megatron mmap dataset prefix (without the .bin/.idx suffix), e.g.:
#   DATA_PATH=/your/datasets/RedPajama-Data-1T-Sample/redpajama-llama2_text_document
DATA_PATH=${DATA_PATH:-<PATH_TO_DATASETS>/RedPajama-Data-1T-Sample/redpajama-llama2_text_document}
DATA_CACHE_PATH=${DATA_CACHE_PATH:-<PATH_TO_DATASETS>/RedPajama-Data-1T-Sample/redpajama-llama2_text_document/index-cache}

source "${SCRIPT_DIR}/env.sh"

TP_SIZE=1
SEQ_LEN=1024
PP_SIZE=1
EP_SIZE=2
MICRO_BATCH=1
GLOBAL_BATCH=2
MODEL_SIZE=${MODEL_SIZE:-"8x7b"}


if [ "${MODEL_SIZE}" = "8x7b" ]; then
    NUM_LAYERS=32
    HIDDEN_SIZE=4096
    FFN_HIDDEN_SIZE=14336
    NUM_ATTENTION_HEADS=32
    NUM_QUERY_GROUPS=8
    MAX_POSITION_EMBEDDINGS=32768
    NUM_EXPERTS=8
    MOE_AUX_LOSS_COEFF=0.02
    VOCAB_SIZE=32000
elif [ "${MODEL_SIZE}" = "8x22b" ]; then
    NUM_LAYERS=56
    HIDDEN_SIZE=6144
    FFN_HIDDEN_SIZE=16384
    NUM_ATTENTION_HEADS=48
    NUM_QUERY_GROUPS=8
    MAX_POSITION_EMBEDDINGS=65536
    NUM_EXPERTS=8
    MOE_AUX_LOSS_COEFF=0.001
    VOCAB_SIZE=32768
else
    echo "Unsupported MODEL_SIZE=${MODEL_SIZE}. Supported values: 8x7b, 8x22b"
    exit 1
fi

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
    --ffn-hidden-size ${FFN_HIDDEN_SIZE}
    --num-attention-heads ${NUM_ATTENTION_HEADS}
    --init-method-std 0.01
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --position-embedding-type rope
    --swiglu
    --untie-embeddings-and-output-weights
    --group-query-attention
    --num-query-groups ${NUM_QUERY_GROUPS}
    --no-masked-softmax-fusion
    --no-position-embedding
    --rotary-base 1000000
    --padded-vocab-size ${VOCAB_SIZE}
    --exit-interval 5
    --no-gradient-accumulation-fusion
    --distributed-timeout-minutes 600
)
#  --profile
#     --use-pytorch-profiler
#     --profile-step-start 1
#     --profile-step-end 3

MOE_ARGS=(
    --num-experts ${NUM_EXPERTS}
    --moe-router-topk 2
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff ${MOE_AUX_LOSS_COEFF}
    --moe-grouped-gemm
    --moe-token-dispatcher-type alltoall
    # --overlap-param-gather
    # --overlap-grad-reduce
)

DATA_ARGS=(
    --tokenizer-type Llama2Tokenizer
    --tokenizer-model ${TOKENIZER_MODEL}
    --data-path $DATA_PATH
    --data-cache-path $DATA_CACHE_PATH
    --split 99990,8,2
)

TRAINING_ARGS=(
    --micro-batch-size ${MICRO_BATCH}
    --global-batch-size ${GLOBAL_BATCH}
    --lr 1e-4
    --train-iters 500000
    --lr-decay-iters 320000
    --lr-decay-style cosine
    --min-lr 1.0e-5
    --weight-decay 0.1
    --lr-warmup-iters 500
    --clip-grad 1.0
    --bf16
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size ${TP_SIZE}
    --pipeline-model-parallel-size ${PP_SIZE}
    --expert-model-parallel-size ${EP_SIZE}
    --use-distributed-optimizer
    --sequence-parallel
)

LOGGING_ARGS=(
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10 \
    # --save $CHECKPOINT_PATH \
    # --load $CHECKPOINT_PATH \
    --tensorboard-dir "${TENSORBOARD_PATH}" \
    --no-load-optim \
    --no-load-rng
)

if [ -n "${WANDB_API_KEY}" ]; then
    LOGGING_ARGS+=(
        --wandb-project ${WANDB_PROJECT:-"Mixtral"}
        --wandb-exp-name ${WANDB_NAME:-"Mixtral_${MODEL_SIZE}"}
    )
fi


torchrun ${DISTRIBUTED_ARGS[@]} model/pretrain_mixtral.py \
    ${MODEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]} \
    2>&1 | tee -a ${LOGS_PATH}/${timestamp}.log
