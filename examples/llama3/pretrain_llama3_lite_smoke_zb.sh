#!/usr/bin/env bash
# Zero Bubble smoke：PP2 / TP1，8 层，GBS16 / MBS4，Flash Attention。
# 用法：bash examples/llama3/pretrain_llama3_lite_smoke_zb.sh
set -eo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "$SCRIPT_DIR"

source /usr/local/birensupa/all/latest/scripts/brsw_set_env.sh
unset BR_UMD_BACKEND BR_UMD_SIM_DEVICE_COUNT TORCH_DEVICE_BACKEND_AUTOLOAD
export NCCL_P2P_DISABLE=1
export NCCL_CUMEM_ENABLE=0
export NCCL_CUMEM_HOST_ENABLE=0
export TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false

mkdir -p "$REPO_ROOT/logs"
SMOKE_RUN_DIR=${SMOKE_RUN_DIR:-$(mktemp -d "$REPO_ROOT/logs/llama3-zb-pp2-l8-gbs16-mbs4-$(date -u +%Y%m%dT%H%M%SZ).XXXXXX")}
mkdir -p "$SMOKE_RUN_DIR"
SMOKE_RUN_DIR=$(cd "$SMOKE_RUN_DIR" && pwd)
export SMOKE_REPO_ROOT="$REPO_ROOT"
export ML_OUTPUT_DIR="$SMOKE_RUN_DIR/output"
export DATA_CACHE_PATH="$SMOKE_RUN_DIR/data-cache"
export EXPNAME=pp2_l8_gbs16_mbs4_zb_flash
export PYTHONPATH="$SMOKE_RUN_DIR:$REPO_ROOT:$REPO_ROOT/third-party/megatron-lm:${PYTHONPATH:-}"

GPUS_PER_NODE=2
NNODES=1
NODE_RANK=0
WORLD_SIZE=2
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-56653}
TP_SIZE=1
PP_SIZE=2
GLOBAL_BATCH=16
MICRO_BATCH=4
SEQ_LEN=64
NLAYERS=8
MODEL_SIZE=lite
HIDDEN_SIZE=128
INTERMEDIATE_SIZE=512
NUM_ATTENTION_HEADS=2
NUM_KEY_VALUE_HEADS=2
VOCAB_SIZE=128256
MAX_POSITION_EMBEDDINGS=512
DATA_PATH=${DATA_PATH:-<PATH_TO_DATASETS>/RedPajama-Data-1T-Sample/redpajama-llama2_text_document}

source "$SCRIPT_DIR/env.sh"
export NCCL_DEBUG_FILE="$ML_OUTPUT_DIR/bccl_debug_%h_%p.log"

# 只替换无参数的 core attention，保留 local QKV/投影/MLP 的 B/W 拆分路径。
cat > "$SMOKE_RUN_DIR/local_flash_spec.py" <<'PY_SPEC'
from megatron.core.extensions.transformer_engine import TEDotProductAttention
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec

layer_spec = get_gpt_layer_local_spec(normalization='RMSNorm')
layer_spec.submodules.self_attention.submodules.core_attention = TEDotProductAttention
PY_SPEC

# 每个 worker 在同一 Python 进程中应用适配、跳过启动预热，再进入原训练入口。
cat > "$SMOKE_RUN_DIR/pretrain_no_warmup.py" <<'PY_WORKER'
import importlib
import os
from pathlib import Path
import runpy
import sys

import megatron_supa  # Apply SUPA adapters before importing the training entrypoint.

initialize = importlib.import_module('megatron.training.initialize')
def skip_startup_warmup(tp_size=None):
    print(f"[rank {os.environ['RANK']}] skipped startup JIT warmup; torch.compile disabled", flush=True)
initialize._warmup_jit_function = skip_startup_warmup

entry = Path(os.environ['SMOKE_REPO_ROOT']) / 'examples/llama3/model/pretrain_llama.py'
sys.argv[0] = str(entry)
runpy.run_path(str(entry), run_name='__main__')
PY_WORKER

DISTRIBUTED_ARGS=(
    --nproc-per-node "$GPUS_PER_NODE"
    --nnodes "$NNODES"
    --node-rank "$NODE_RANK"
    --master-addr "$MASTER_ADDR"
    --master-port "$MASTER_PORT"
    --max-restarts 0
)

MODEL_ARGS=(
    --disable-bias-linear
    --seq-length "$SEQ_LEN"
    --max-position-embeddings "$MAX_POSITION_EMBEDDINGS"
    --num-layers "$NLAYERS"
    --hidden-size "$HIDDEN_SIZE"
    --ffn-hidden-size "$INTERMEDIATE_SIZE"
    --num-attention-heads "$NUM_ATTENTION_HEADS"
    --group-query-attention --num-query-groups "$NUM_KEY_VALUE_HEADS"
    --init-method-std 0.01 --attention-dropout 0.0 --hidden-dropout 0.0
    --normalization RMSNorm --no-persist-layer-norm
    --position-embedding-type rope --swiglu
    --untie-embeddings-and-output-weights --no-position-embedding
    --rotary-base 500000 --padded-vocab-size "$VOCAB_SIZE"
    --transformer-impl local --spec local_flash_spec layer_spec
    --enable-zero-bubble
    # 梯度累积融合默认开启：不能添加 --no-gradient-accumulation-fusion。
    --no-bias-dropout-fusion --attention-backend flash
    --exit-interval 2 --distributed-timeout-minutes 60 --disable-gloo-process-groups
)
DATA_ARGS=(
    --tokenizer-type HuggingFaceTokenizer --tokenizer-model "$TOKENIZER_MODEL"
    --data-path "$DATA_PATH" --data-cache-path "$DATA_CACHE_PATH" --split 99990,8,2
)
TRAINING_ARGS=(
    --micro-batch-size "$MICRO_BATCH" --global-batch-size "$GLOBAL_BATCH"
    --lr 1e-4 --train-iters 4 --lr-decay-iters 4 --lr-decay-style cosine
    --min-lr 1.0e-5 --weight-decay 0.1 --lr-warmup-iters 1 --clip-grad 1.0 --bf16
)
MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size "$TP_SIZE" --pipeline-model-parallel-size "$PP_SIZE"
    --expert-model-parallel-size 1
)
LOGGING_ARGS=(
    --log-interval 1 --save-interval 1000 --eval-interval 1000 --eval-iters 1
    --tensorboard-dir "$TENSORBOARD_PATH" --no-load-optim --no-load-rng
)

torchrun "${DISTRIBUTED_ARGS[@]}" "$SMOKE_RUN_DIR/pretrain_no_warmup.py" \
    "${MODEL_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${MODEL_PARALLEL_ARGS[@]}" \
    "${LOGGING_ARGS[@]}" \
    2>&1 | tee -a "$LOGS_PATH/$timestamp.log"
