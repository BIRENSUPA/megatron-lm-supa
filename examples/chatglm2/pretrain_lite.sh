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
GLOBAL_BATCH=2
MICRO_BATCH=1
SEQ_LEN=128
MODEL_SIZE=lite

export CUDA_VISIBLE_DEVICES=0,1
source "${SCRIPT_DIR}/env.sh"

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

# ChatGLM2-6B Configuration
# Reference: https://huggingface.co/zai-org/chatglm2-6b/blob/main/config.json
# - Hidden size: 4096
# - Attention heads: 32
# - KV heads (MQA): 2
# - Layers: 28
# - FFN hidden size: 13696
# - Vocab size (padded): 65024
# - Max sequence length: 32768
# - Normalization: RMSNorm
# - Position embedding: RoPE (original_rope=true)

if [ $MODEL_SIZE = lite ]; then
    NUM_LAYERS=1
    HIDDEN_SIZE=128
    NUM_ATTENTION_HEADS=2
    INTERMEDIATE_SIZE=1024
    NUM_KEY_VALUE_HEADS=2
    VOCAB_SIZE=65024
    GQA_OPTIONS=" \
                --group-query-attention \
                --num-query-groups ${NUM_KEY_VALUE_HEADS}"

    TIE_OPTIONS=""
fi

MODEL_ARGS=(
    --disable-bias-linear
    --seq-length ${SEQ_LEN}
    --max-position-embeddings 32768
    --num-layers ${NUM_LAYERS}
    --hidden-size ${HIDDEN_SIZE}
    --ffn-hidden-size ${INTERMEDIATE_SIZE}
    --num-attention-heads ${NUM_ATTENTION_HEADS}
    --init-method-std 0.01
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --position-embedding-type rope
    --swiglu
    --untie-embeddings-and-output-weights
    --no-masked-softmax-fusion
    --no-position-embedding
    --rotary-base 10000
    --no-persist-layer-norm
    --padded-vocab-size ${VOCAB_SIZE}
    --transformer-impl transformer_engine

    --no-gradient-accumulation-fusion
    --no-bias-dropout-fusion
    --exit-interval 5
    --distributed-timeout-minutes 1000
    --attention-backend fused
)

DATA_ARGS=(
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model $TOKENIZER_PATH
    --vocab-file $TOKENIZER_PATH/tokenizer.model
    --trust-remote-code
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
        --wandb-project ${WANDB_PROJECT:-"ChatGLM2"}
        --wandb-exp-name ${WANDB_NAME:-"ChatGLM2_6B"}
    )
fi


torchrun ${DISTRIBUTED_ARGS[@]} model/pretrain_chatglm2.py \
    ${MODEL_ARGS[@]} \
    ${GQA_OPTIONS} \
    ${TIE_OPTIONS} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]} \
    2>&1 | tee -a ${LOGS_PATH}/${timestamp}.log
