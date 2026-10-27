#!/bin/bash


SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${SCRIPT_DIR}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third-party/megatron-lm:${PYTHONPATH}"

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6001"}
NNODES=${SLURM_NNODES:-"1"}
NODE_RANK=${RANK:-"0"}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

# DATA_PATH is the Megatron mmap dataset prefix (without the .bin/.idx suffix), e.g.:
#   DATA_PATH=/your/datasets/RedPajama-Data-1T-Sample/redpajama-llama2_text_document
DATA_PATH=${DATA_PATH:-<PATH_TO_DATASETS>/Qwen/qwen-datasets/mmap_qwen3_datasets_text_document}
DATA_CACHE_PATH=${DATA_CACHE_PATH:-<PATH_TO_DATASETS>/Qwen/qwen-datasets/mmap_qwen3_datasets_text_document}

TP_SIZE=8
CP_SIZE=1
PP_SIZE=1
GLOBAL_BATCH=128
MICRO_BATCH=1
SEQ_LEN=32768
MODEL_SIZE=32B

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
source "${SCRIPT_DIR}/env.sh"

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

if [ $MODEL_SIZE = 0.6B ]; then
    NUM_LAYERS=28
    HIDDEN_SIZE=1024
    NUM_ATTENTION_HEADS=16
    INTERMEDIATE_SIZE=3072
    NUM_KEY_VALUE_HEADS=8
    VOCAB_SIZE=151936
    GQA_OPTIONS=" \
                --group-query-attention \
                --num-query-groups ${NUM_KEY_VALUE_HEADS}"

    TIE_OPTIONS=""
elif [ $MODEL_SIZE = 1.7B ]; then
    NUM_LAYERS=28
    HIDDEN_SIZE=2048
    NUM_ATTENTION_HEADS=16
    INTERMEDIATE_SIZE=6144
    NUM_KEY_VALUE_HEADS=8
    VOCAB_SIZE=151936
    GQA_OPTIONS=" \
                --group-query-attention \
                --num-query-groups ${NUM_KEY_VALUE_HEADS}"

    TIE_OPTIONS=""
elif [ $MODEL_SIZE = 4B ]; then
    NUM_LAYERS=36
    HIDDEN_SIZE=2560
    NUM_ATTENTION_HEADS=32
    INTERMEDIATE_SIZE=9728
    NUM_KEY_VALUE_HEADS=8
    VOCAB_SIZE=151936
    GQA_OPTIONS=" \
                --group-query-attention \
                --num-query-groups ${NUM_KEY_VALUE_HEADS}"

    TIE_OPTIONS=""

elif [ $MODEL_SIZE = 8B ]; then
    NUM_LAYERS=36
    HIDDEN_SIZE=4096
    NUM_ATTENTION_HEADS=32
    INTERMEDIATE_SIZE=12288
    NUM_KEY_VALUE_HEADS=8
    VOCAB_SIZE=151936
    GQA_OPTIONS=" \
                --group-query-attention \
                --num-query-groups ${NUM_KEY_VALUE_HEADS}"

    TIE_OPTIONS=" \
            --untie-embeddings-and-output-weights \
            "
elif [ $MODEL_SIZE = 14B ]; then
    NUM_LAYERS=40
    HIDDEN_SIZE=5120
    NUM_ATTENTION_HEADS=40
    INTERMEDIATE_SIZE=17408
    NUM_KEY_VALUE_HEADS=8
    VOCAB_SIZE=151936
    GQA_OPTIONS=" \
                --group-query-attention \
                --num-query-groups ${NUM_KEY_VALUE_HEADS}"

    TIE_OPTIONS=" \
            --untie-embeddings-and-output-weights \
            "
elif [ $MODEL_SIZE = 32B ]; then
    NUM_LAYERS=64
    HIDDEN_SIZE=5120
    NUM_ATTENTION_HEADS=64
    INTERMEDIATE_SIZE=25600
    NUM_KEY_VALUE_HEADS=8
    VOCAB_SIZE=151936
    GQA_OPTIONS=" \
                --group-query-attention \
                --num-query-groups ${NUM_KEY_VALUE_HEADS}"

    TIE_OPTIONS=" \
            --untie-embeddings-and-output-weights \
            "
fi

MODEL_ARGS=(
    --disable-bias-linear
    --seq-length ${SEQ_LEN}
    --max-position-embeddings 40960 
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
    --no-masked-softmax-fusion
    --no-position-embedding
    --rotary-base 1000000
    --qk-layernorm
    --no-persist-layer-norm
    --padded-vocab-size ${VOCAB_SIZE}
    #--transformer-impl local
    --transformer-impl transformer_engine
    --no-bias-dropout-fusion
    --no-gradient-accumulation-fusion
    --exit-interval 5 
    --distributed-timeout-minutes 1000
    #--attention-backend flash
    --attention-backend fused 
    --profile
    --use-pytorch-profiler
    --profile-step-start 1
    --profile-step-end 2
    --pytorch-profiler-collect-shapes
)
#  --profile
#     --use-pytorch-profiler
#     --profile-step-start 1
#     --profile-step-end 2

    #--merge-file ${MERGE_FILE}
DATA_ARGS=(
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model "${TOKENIZER_MODEL}"
    --data-path $DATA_PATH
    --data-cache-path $DATA_CACHE_PATH
    --split 99990,8,2
    --no-create-attention-mask-in-dataloader
    --num-workers 1
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
    #--enable-zero-bubble
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size ${TP_SIZE}
    --pipeline-model-parallel-size ${PP_SIZE}
    --context-parallel-size ${CP_SIZE}
    --expert-model-parallel-size 1
    --use-distributed-optimizer
    --sequence-parallel
)

LOGGING_ARGS=(
    --log-interval 1
    --save-interval 10000
    --eval-interval 1000
    --eval-iters 10
    #--save $CHECKPOINT_PATH \
    #--load $CHECKPOINT_PATH \
    --tensorboard-dir "${TENSORBOARD_PATH}"
    --no-load-optim
    --no-load-rng
    --log-throughput
)

if [ -n "${WANDB_API_KEY}" ]; then
    LOGGING_ARGS+=(
        --wandb-project ${WANDB_PROJECT:-"Mixtral"}
        --wandb-exp-name ${WANDB_NAME:-"Mixtral_8x7B"}
    )
fi


torchrun ${DISTRIBUTED_ARGS[@]} model/pretrain_qwen.py \
    ${MODEL_ARGS[@]} \
    ${GQA_OPTIONS} \
    ${TIE_OPTIONS} \
    ${MOE_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]} \
    2>&1 | tee -a ${LOGS_PATH}/${timestamp}.log
