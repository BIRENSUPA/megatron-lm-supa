#!/bin/bash
ulimit -n 65536

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
DATA_PATH=${DATA_PATH:-<PATH_TO_DATASETS>/Qwen/qwen-datasets/mmap_qwen3_datasets_text_document}
DATA_CACHE_PATH=${DATA_CACHE_PATH:-<PATH_TO_DATASETS>/Qwen/qwen-datasets/mmap_qwen3_datasets_text_document}

TP_SIZE=1
PP_SIZE=1
GLOBAL_BATCH=2
MICRO_BATCH=1
NLAYERS=48
MODEL_SIZE=moe_30b

export SUPA_VISIBLE_DEVICES=0,1
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
    --seq-length 4096 
    --max-position-embeddings 40960 
    --num-layers ${NLAYERS}
    --hidden-size 2048
    --ffn-hidden-size 6144 
    --num-attention-heads 32 
    --init-method-std 0.008
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --use-rotary-position-embeddings
    --position-embedding-type rope
    --swiglu
    --untie-embeddings-and-output-weights
    --group-query-attention
    --num-query-groups 4
    --kv-channels 128
    --no-masked-softmax-fusion
    --no-position-embedding
    --rotary-base 1000000
    --qk-layernorm
    --no-persist-layer-norm
    --padded-vocab-size 151936
    --transformer-impl transformer_engine
    --no-gradient-accumulation-fusion
    --no-bias-dropout-fusion
    --exit-interval 5 
)
#  --profile
#     --use-pytorch-profiler
#     --tensorboard-dir "./prof"
#     --profile-step-start 1
#     --profile-step-end 2

    #--merge-file ${MERGE_FILE}
MOE_ARGS=(
    --num-experts 128
    --moe-router-topk 8
    --moe-ffn-hidden-size 768
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 1e-3
    --moe-grouped-gemm
    --moe-token-dispatcher-type alltoall
    --overlap-grad-reduce
)

DATA_ARGS=(
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model "${TOKENIZER_MODEL}"
    --data-path $DATA_PATH
    --data-cache-path $DATA_CACHE_PATH
    --split 99,1,0
    --num-workers 1
)

TRAINING_ARGS=(
    --transformer-impl transformer_engine
    --cross-entropy-loss-fusion
    --micro-batch-size ${MICRO_BATCH}
    --global-batch-size ${GLOBAL_BATCH}
    --lr 1e-5
    --train-iters 1
    --lr-decay-iters 320000
    --lr-decay-style cosine
    --min-lr 1.0e-6
    --weight-decay 0.01
    --adam-beta1 0.9
    --adam-beta2 0.95
    --lr-warmup-iters 500
    --clip-grad 1.0
    --bf16
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size ${TP_SIZE}
    --pipeline-model-parallel-size ${PP_SIZE}
    --expert-model-parallel-size 2
    # --use-distributed-optimizer
    --sequence-parallel
)

LOGGING_ARGS=(
    --log-interval 1 \
    --log-throughput \
    --save-interval 10000 \
    #--eval-interval 10000 \
    --eval-iters 0 \
    #--save $CHECKPOINT_PATH \
    #--load $CHECKPOINT_PATH \
    #--tensorboard-dir "${CHECKPOINT_PATH}/tensorboard" \
    --no-load-optim \
    --no-load-rng \
    --no-save-optim \
    --ckpt-format torch_dist
)

if [ -n "${WANDB_API_KEY}" ]; then
    LOGGING_ARGS+=(
        --wandb-project ${WANDB_PROJECT:-"Qwen3"}
        --wandb-exp-name ${WANDB_NAME:-"Qwen3_moe"}
    )
fi


torchrun ${DISTRIBUTED_ARGS[@]} model/pretrain_qwen.py \
    ${MODEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]} \
    --distributed-timeout-minutes 600 \
    2>&1 | tee -a ${LOGS_PATH}/${timestamp}.log
