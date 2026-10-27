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

TP_SIZE=2
PP_SIZE=1
CP_SIZE=4
GLOBAL_BATCH=128
MICRO_BATCH=1
SEQ_LEN=32768
MODEL_SIZE=72B

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
source "${SCRIPT_DIR}/env.sh"

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

if [ $MODEL_SIZE = 0.5B ]; then
    NUM_LAYERS=24
    HIDDEN_SIZE=896
    NUM_ATTENTION_HEADS=14
    INTERMEDIATE_SIZE=4864
    NUM_KEY_VALUE_HEADS=2
    RMS_NORM_EPS=1e-6
    GQA_OPTIONS=" \
                --group-query-attention \
                --num-query-groups ${NUM_KEY_VALUE_HEADS}"
    TIE_OPTIONS=""
elif [ $MODEL_SIZE = 1.5B ]; then
    NUM_LAYERS=28
    HIDDEN_SIZE=1536
    NUM_ATTENTION_HEADS=12
    INTERMEDIATE_SIZE=8960
    NUM_KEY_VALUE_HEADS=2
    RMS_NORM_EPS=1e-6
    GQA_OPTIONS=" \
                --group-query-attention \
                --num-query-groups ${NUM_KEY_VALUE_HEADS}"
    TIE_OPTIONS=""
elif [ $MODEL_SIZE = 3B ]; then
    NUM_LAYERS=36
    HIDDEN_SIZE=2048
    NUM_ATTENTION_HEADS=16
    INTERMEDIATE_SIZE=11008
    NUM_KEY_VALUE_HEADS=2
    RMS_NORM_EPS=1e-6
    GQA_OPTIONS=" \
                --group-query-attention \
                --num-query-groups ${NUM_KEY_VALUE_HEADS}"
    TIE_OPTIONS=""
elif [ $MODEL_SIZE = 7B ]; then
    NUM_LAYERS=28
    HIDDEN_SIZE=3584
    NUM_ATTENTION_HEADS=28
    INTERMEDIATE_SIZE=18944
    NUM_KEY_VALUE_HEADS=4
    RMS_NORM_EPS=1e-6
    GQA_OPTIONS=" \
                --group-query-attention \
                --num-query-groups ${NUM_KEY_VALUE_HEADS}"
    TIE_OPTIONS=" \
                --untie-embeddings-and-output-weights"
elif [ $MODEL_SIZE = 14B ]; then
    NUM_LAYERS=48
    HIDDEN_SIZE=5120
    NUM_ATTENTION_HEADS=40
    INTERMEDIATE_SIZE=13824
    NUM_KEY_VALUE_HEADS=8
    RMS_NORM_EPS=1e-5
    GQA_OPTIONS=" \
                --group-query-attention \
                --num-query-groups ${NUM_KEY_VALUE_HEADS}"
    TIE_OPTIONS=" \
                --untie-embeddings-and-output-weights"
elif [ $MODEL_SIZE = 32B ]; then
    NUM_LAYERS=64
    HIDDEN_SIZE=5120
    NUM_ATTENTION_HEADS=40
    INTERMEDIATE_SIZE=27648
    NUM_KEY_VALUE_HEADS=8
    RMS_NORM_EPS=1e-5
    GQA_OPTIONS=" \
                --group-query-attention \
                --num-query-groups ${NUM_KEY_VALUE_HEADS}"
    TIE_OPTIONS=" \
                --untie-embeddings-and-output-weights"
elif [ $MODEL_SIZE = 72B ]; then
    NUM_LAYERS=20
    HIDDEN_SIZE=8192
    NUM_ATTENTION_HEADS=64
    INTERMEDIATE_SIZE=29568
    NUM_KEY_VALUE_HEADS=8
    RMS_NORM_EPS=1e-5
    GQA_OPTIONS=" \
                --group-query-attention \
                --num-query-groups ${NUM_KEY_VALUE_HEADS}"
    TIE_OPTIONS=" \
                --untie-embeddings-and-output-weights"
fi

VOCAB_SIZE=152064
MODEL_ARGS=(
    --disable-bias-linear
    --seq-length ${SEQ_LEN}
    --max-position-embeddings 131072
    --num-layers ${NUM_LAYERS}
    --hidden-size ${HIDDEN_SIZE}
    --ffn-hidden-size ${INTERMEDIATE_SIZE}
    --num-attention-heads ${NUM_ATTENTION_HEADS}
    --init-method-std 0.008
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --norm-epsilon ${RMS_NORM_EPS}
    --position-embedding-type rope
    --swiglu
    --no-masked-softmax-fusion
    --no-position-embedding
    --use-rotary-position-embeddings
    --rotary-percent 1.0
    --rotary-base 1000000
    --add-qkv-bias
    --padded-vocab-size ${VOCAB_SIZE}
    --transformer-impl transformer_engine
    --no-gradient-accumulation-fusion
    --no-bias-dropout-fusion
    --distributed-timeout-minutes 1000
    --attention-backend fused
    --exit-interval 5
)

DATA_ARGS=(
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model "${TOKENIZER_MODEL}"
    --data-path $DATA_PATH
    --data-cache-path $DATA_CACHE_PATH
    --split 99,1,0
    --no-create-attention-mask-in-dataloader
    --num-workers 1
)

TRAINING_ARGS=(
    --micro-batch-size ${MICRO_BATCH}
    --global-batch-size ${GLOBAL_BATCH}
    --lr 1e-4
    --train-iters 500
    --lr-decay-iters 400
    --lr-decay-style cosine
    --min-lr 1e-5
    --weight-decay 0.1
    --lr-warmup-iters 50
    --clip-grad 1.0
    --bf16
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
    --eval-iters 1
    --tensorboard-dir "${TENSORBOARD_PATH}"
    --no-load-optim
    --no-load-rng
    --log-throughput
    --profile
    --use-pytorch-profiler
    --profile-step-start 1
    --profile-step-end 2
    --pytorch-profiler-collect-shapes
)

torchrun ${DISTRIBUTED_ARGS[@]} model/pretrain_qwen.py \
    ${MODEL_ARGS[@]} \
    ${GQA_OPTIONS} \
    ${TIE_OPTIONS} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]} \
    2>&1 | tee -a ${LOGS_PATH}/${timestamp}.log
