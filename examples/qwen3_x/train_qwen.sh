#!/bin/bash
set -euo pipefail

####################################################
#
# Usage:
#   bash train_qwen.sh [OPTIONS]
#
# OPTIONS 分为四类配置：
#
# 1. Torchrun 配置（分布式启动参数）
#    --master_addr ADDR        Master 节点地址（默认: localhost）
#    --master_port PORT        Master 节点监听端口（默认: 19981）
#    --node_num N              分布式作业的总节点数（默认: 1）
#    --node_rank R             当前节点的索引（从 0 开始， 默认: 0）
#    --nproc_per_node P        每个节点使用的 GPU/DCU 数量（默认: 8）
#
# 2. 模型选择
#    --model_size SIZE         模型规模：27B（Qwen3.6-27B dense）或 qwen3.5: {35B-A3B} 或 122B（Qwen3.5-122B-A10B MoE）（默认: 27B）
#
# 3. 并行配置（用于控制模型并行策略）
#    --tp TP                   Tensor Parallel（张量并行）（默认: 8）
#    --pp PP                   Pipeline Parallel（流水线并行）（默认: 4）
#    --cp CP                   Context Parallel（上下文并行，当前实现不支持 >1）（默认: 1）
#    --ep EP                   Expert Parallel（专家并行，仅 122B MoE 使用）（默认: 2）
#    --etp ETP                 Expert Tensor Parallel（专家张量并行，仅 122B MoE 使用）（默认: 1）
#
# 4. 其他训练配置
#    --mbs MBS                 Megatron 的 Micro Batch Size（默认: 1）
#    --mbn MBN                 Megatron 的 Micro Batch Num（默认: 64）
#    --seq_len LEN             训练时的序列长度（默认: 32768）
#    --train_iters IT          总训练迭代次数（默认: 30）
#    --precision PREC          训练数值精度（仅支持 bf16）（默认: bf16）
#    --checkpoint_path PATH    检查点存储路径（默认: 不使用检查点）
#    --data_source PATH        训练数据路径（Megatron 数据前缀；或设置 DATA_PATH）
#    --tokenizer_path PATH     tokenizer 目录（与模型匹配；或设置 TOKENIZER_PATH）
#
# 示例：
#    - 4 节点训练 Qwen3.6-27B（每台节点运行，--node_rank 要变）：
#        bash train_qwen.sh --master_addr 192.168.1.10 --node_num 4 --node_rank ${RANK} \
#          --model_size 27B --tp 8 --pp 4 --cp 1 --mbs 1 --mbn 128 --seq_len 32768 --precision bf16
#    - 8 节点训练 Qwen3.5-122B-A10B（MoE，--ep/--etp 按专家并行策略覆盖）：
#        bash train_qwen.sh --master_addr 192.168.1.10 --node_num 8 --node_rank ${RANK} \
#          --model_size 122B --tp 8 --pp 4 --cp 1 --ep 2 --etp 1 --mbs 1 --mbn 64 --seq_len 32768 --precision bf16
#
# 注意：
#   - 所有选项均为可选，未指定时使用默认值。
#   - --cp 当前只支持 1：Qwen3.5/3.6 使用 gated_delta_net，Megatron 实现不支持 cp>1。
#   - --ep/--etp 的拓扑合法性由 Megatron 自身校验，本脚本仅透传。
#
# supported model size:
#	  - qwen3.6: {27B}        (dense)
#	  - qwen3.5: {35B-A3B}  (sparse/MoE)
#	  - qwen3.5: {122B-A10B}  (sparse/MoE)
####################################################

export CUDA_DEVICE_MAX_CONNECTIONS=1
# export HIPBLASLT_ALLOW_TF32=${HIPBLASLT_ALLOW_TF32:-1}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 通过同级 wrap_pretrain_gpt.py 启动：在 import megatron 之前激活 megatron_supa patcher
ENTRY_SCRIPT="${SCRIPT_DIR}/wrap_pretrain_gpt.py"

# ----------------------------------------------------------------------------
# Torchrun / 分布式启动参数
master_addr="${master_addr:-localhost}"
master_port="${master_port:-19981}"
node_num="${node_num:-1}"
node_rank="${node_rank:-0}"
nproc_per_node="${nproc_per_node:-8}"

# 模型选择
model_size="${model_size:-27B}"

# 并行配置
tp="${tp:-8}"
pp="${pp:-4}"
cp="${cp:-1}"
ep="${ep:-2}"
etp="${etp:-1}"

# 训练配置
mbs="${mbs:-1}"
mbn="${mbn:-64}"    
seq_len="${seq_len:-32768}"
train_iters="${train_iters:-5}"
precision="${precision:-bf16}"

# 数据和 tokenizer 路径通过环境变量或命令行显式传入。
checkpoint_path="${checkpoint_path:-}"
data_source="${data_source:-${DATA_PATH:-}}"
tokenizer_path="${tokenizer_path:-${TOKENIZER_PATH:-}}"

# 命令行解析：同名 --flag 覆盖上述默认值
while [[ $# -gt 0 ]]; do
  case "$1" in
    --master_addr) master_addr="$2"; shift 2;;
    --master_port) master_port="$2"; shift 2;;
    --node_num) node_num="$2"; shift 2;;
    --node_rank) node_rank="$2"; shift 2;;
    --nproc_per_node) nproc_per_node="$2"; shift 2;;
    --model_size) model_size="$2"; shift 2;;
    --tp) tp="$2"; shift 2;;
    --pp) pp="$2"; shift 2;;
    --cp) cp="$2"; shift 2;;
    --ep) ep="$2"; shift 2;;
    --etp) etp="$2"; shift 2;;
    --mbs) mbs="$2"; shift 2;;
    --mbn) mbn="$2"; shift 2;;
    --seq_len) seq_len="$2"; shift 2;;
    --train_iters) train_iters="$2"; shift 2;;
    --precision) precision="$2"; shift 2;;
    --checkpoint_path) checkpoint_path="$2"; shift 2;;
    --data_source) data_source="$2"; shift 2;;
    --tokenizer_path) tokenizer_path="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

export TORCH_COMPILE_DISABLE=1

# Model settings：按 model_size 设置模型常量与 MoE 参数
MOE_ARGS=""
LOG_NAME_SUFFIX=""
case "${model_size:-27B}" in
  27|27B)
    # Qwen3.6-27B dense config
    MODEL_NAME="qwen3.6-27B"
    NUM_LAYERS=64
    HIDDEN_SIZE=5120
    FFN_HIDDEN_SIZE=17408
    NUM_ATTN_HEADS=24
    NUM_KV_HEADS=4
    VOCAB_SIZE=248320
    KV_CHANNELS=256
    ROPE_THETA=10000000
    ROTARY_PERCENT=0.25
    RMS_NORM_EPS=1e-6
    LINEAR_ATTN_FREQ=4
    LINEAR_CONV_KERNEL_DIM=4
    LINEAR_KEY_HEAD_DIM=128
    LINEAR_VALUE_HEAD_DIM=128
    LINEAR_NUM_KEY_HEADS=16
    LINEAR_NUM_VALUE_HEADS=48
    GATED_DELTA_NET_LABEL="Qwen3.6"
    ;;
  122|122B)
    # Qwen3.5-122B-A10B MoE config
    MODEL_NAME="qwen3.5-122B-A10B"
    NUM_LAYERS=48
    HIDDEN_SIZE=3072
    FFN_HIDDEN_SIZE=1024
    NUM_ATTN_HEADS=32
    NUM_KV_HEADS=2
    VOCAB_SIZE=248320
    KV_CHANNELS=256
    ROPE_THETA=10000000
    ROTARY_PERCENT=0.25
    RMS_NORM_EPS=1e-6
    LINEAR_ATTN_FREQ=4
    LINEAR_CONV_KERNEL_DIM=4
    LINEAR_KEY_HEAD_DIM=128
    LINEAR_VALUE_HEAD_DIM=128
    LINEAR_NUM_KEY_HEADS=16
    LINEAR_NUM_VALUE_HEADS=64
    ROUTER_TOPK=8
    NUM_EXPERTS=256
    MOE_INTERMEDIATE_SIZE=1024
    # 绕过 122B gated_delta_net 路径上的 Dynamo 编译兼容问题
    export TORCHDYNAMO_DISABLE=${TORCHDYNAMO_DISABLE:-1}
    export TORCH_COMPILE_DISABLE=${TORCH_COMPILE_DISABLE:-1}
    MOE_ARGS=" \
      --moe-grouped-gemm \
      --moe-token-dispatcher-type alltoall \
      --moe-router-topk ${ROUTER_TOPK} \
      --num-experts ${NUM_EXPERTS} \
      --expert-tensor-parallel-size ${etp} \
      --expert-model-parallel-size ${ep} \
      --moe-ffn-hidden-size ${MOE_INTERMEDIATE_SIZE} \
      --moe-router-load-balancing-type aux_loss \
      --moe-aux-loss-coeff 0.001 \
      --moe-layer-freq '([1]*${NUM_LAYERS})' \
    "
    LOG_NAME_SUFFIX="_ep${ep}_etp${etp}"
    GATED_DELTA_NET_LABEL="Qwen3.5"
    ;;
  35|35B|35B-A3B)
    # Qwen3.5-35B-A3B MoE config (config.json: Qwen3_5MoeForConditionalGeneration, text_config)
    MODEL_NAME="qwen3.5-35B-A3B"
    NUM_LAYERS=40                # text_config.num_hidden_layers
    HIDDEN_SIZE=2048             # text_config.hidden_size
    FFN_HIDDEN_SIZE=512          # text_config.moe_intermediate_size (per-expert FFN)
    NUM_ATTN_HEADS=16            # text_config.num_attention_heads
    NUM_KV_HEADS=2               # text_config.num_key_value_heads
    VOCAB_SIZE=248320            # text_config.vocab_size
    KV_CHANNELS=256              # text_config.head_dim
    ROPE_THETA=10000000          # rope_parameters.rope_theta
    ROTARY_PERCENT=0.25          # rope_parameters.partial_rotary_factor
    RMS_NORM_EPS=1e-6            # text_config.rms_norm_eps
    LINEAR_ATTN_FREQ=4           # text_config.full_attention_interval
    LINEAR_CONV_KERNEL_DIM=4     # text_config.linear_conv_kernel_dim
    LINEAR_KEY_HEAD_DIM=128      # text_config.linear_key_head_dim
    LINEAR_VALUE_HEAD_DIM=128    # text_config.linear_value_head_dim
    LINEAR_NUM_KEY_HEADS=16      # text_config.linear_num_key_heads
    LINEAR_NUM_VALUE_HEADS=32    # text_config.linear_num_value_heads
    ROUTER_TOPK=8                # text_config.num_experts_per_tok
    NUM_EXPERTS=256              # text_config.num_experts
    MOE_INTERMEDIATE_SIZE=512    # text_config.moe_intermediate_size
    # 绕过 35B gated_delta_net 路径上的 Dynamo 编译兼容问题 (与 122B 分支一致)
    export TORCHDYNAMO_DISABLE=${TORCHDYNAMO_DISABLE:-1}
    export TORCH_COMPILE_DISABLE=${TORCH_COMPILE_DISABLE:-1}
    MOE_ARGS=" \
      --moe-grouped-gemm \
      --moe-token-dispatcher-type alltoall \
      --moe-router-topk ${ROUTER_TOPK} \
      --num-experts ${NUM_EXPERTS} \
      --expert-tensor-parallel-size ${etp} \
      --expert-model-parallel-size ${ep} \
      --moe-ffn-hidden-size ${MOE_INTERMEDIATE_SIZE} \
      --moe-router-load-balancing-type aux_loss \
      --moe-aux-loss-coeff 0.001 \
      --moe-layer-freq '([1]*${NUM_LAYERS})' \
    "
    LOG_NAME_SUFFIX="_ep${ep}_etp${etp}"
    GATED_DELTA_NET_LABEL="Qwen3.5"
    ;;
  *)
    echo "NotImplementedError: model_size ${model_size:-} not supported. Supported: 27B, 122B, 35B." >&2
    exit 2
    ;;
esac

      # --moe-shared-expert-intermediate-size ${MOE_INTERMEDIATE_SIZE} \
      # --moe-shared-expert-gate \
      # --attention-output-gate \

if [[ "${cp}" != "1" ]]; then
  echo "${GATED_DELTA_NET_LABEL} gated_delta_net currently only supports --cp 1." >&2
  exit 2
fi

if [[ "${precision}" == "bf16" ]]; then
  PRECISION_ARGS="--bf16"
  DT="bf16"
else
  echo "NotImplementedError: precision ${precision} not supported. Supported: bf16." >&2
  exit 2
fi

: "${data_source:?Set DATA_PATH or pass --data_source}"
: "${tokenizer_path:?Set TOKENIZER_PATH or pass --tokenizer_path}"
DATASET_ARGS="--data-path ${data_source}"
DATASET_ARGS+=" --num-workers 16"

TOKENIZER_ARGS="--tokenizer-type HuggingFaceTokenizer --tokenizer-model ${tokenizer_path}"

if [[ -z "${checkpoint_path}" ]]; then
  CHECKPOINT_ARGS=""
else
  CHECKPOINT_ARGS=" \
  --load ${checkpoint_path} \
  --save ${checkpoint_path} \
  --no-load-optim \
  --no-load-rng \
  "
fi

WORLD_SIZE=$((${nproc_per_node} * ${node_num}))
MODEL_PARALLEL_SIZE=$((${tp} * ${pp} * ${cp}))
if (( WORLD_SIZE % MODEL_PARALLEL_SIZE != 0 )); then
  echo "WORLD_SIZE=${WORLD_SIZE} must be divisible by tp*pp*cp=${MODEL_PARALLEL_SIZE}." >&2
  exit 2
fi
DP=$((${WORLD_SIZE} / ${MODEL_PARALLEL_SIZE}))
GLOBAL_BATCH_SIZE=$((${DP} * ${mbs} * ${mbn}))

DISTRIBUTED_ARGS=" \
  --nproc_per_node ${nproc_per_node} \
  --nnodes ${node_num} \
  --node_rank ${node_rank} \
  --master_addr ${master_addr} \
  --master_port ${master_port} \
"

COMM_ARGS=""
OPTIMIZER_ARGS=""
if [[ "${DP}" -gt 1 ]]; then
  COMM_ARGS="--overlap-grad-reduce --overlap-param-gather"
  OPTIMIZER_ARGS="--use-distributed-optimizer"
fi

# 输出目录：同级 ${MODEL_NAME}-${WORLD_SIZE}-<并行配置>/logs（不依赖绝对路径）
# 必须在 CMD 构造之前定义，因为 --tensorboard-dir 引用了 ${OUTPUT_DIR}
PARALLEL_TAG="tp${tp}_pp${pp}_cp${cp}_dp${DP}${LOG_NAME_SUFFIX}"
OUTPUT_DIR="${SCRIPT_DIR}/${MODEL_NAME}-card${WORLD_SIZE}-${PARALLEL_TAG}"
LOG_DIR="${OUTPUT_DIR}/logs"
LOG_FILE_NAME="megatron_${MODEL_NAME}_seq${seq_len}_tp${tp}_pp${pp}_cp${cp}_dp${DP}${LOG_NAME_SUFFIX}_mb${mbs}_gb${GLOBAL_BATCH_SIZE}_${DT}_NODE${node_rank}.log"
LOG_FILE="${LOG_DIR}/${LOG_FILE_NAME}"

CMD="torchrun ${DISTRIBUTED_ARGS} \
  ${ENTRY_SCRIPT} \
  --use-mcore-models \
  --no-gradient-accumulation-fusion \
  --recompute-granularity full \
  --recompute-method uniform \
  --recompute-num-layers 1 \
  ${COMM_ARGS} \
  --tensor-model-parallel-size ${tp} \
  --pipeline-model-parallel-size ${pp} \
  --context-parallel-size ${cp} \
  --sequence-parallel \
  --num-layers ${NUM_LAYERS} \
  --hidden-size ${HIDDEN_SIZE} \
  --ffn-hidden-size ${FFN_HIDDEN_SIZE} \
  --num-attention-heads ${NUM_ATTN_HEADS} \
  --kv-channels ${KV_CHANNELS} \
  --vocab-size ${VOCAB_SIZE} \
  --group-query-attention \
  --num-query-groups ${NUM_KV_HEADS} \
  --qk-layernorm \
  --hidden-dropout 0.0 \
  --attention-dropout 0 \
  --swiglu \
  --disable-bias-linear \
  --untie-embeddings-and-output-weights \
  --micro-batch-size ${mbs} \
  --global-batch-size ${GLOBAL_BATCH_SIZE} \
  --seq-length ${seq_len} \
  --max-position-embeddings ${seq_len} \
  --normalization RMSNorm \
  --norm-epsilon ${RMS_NORM_EPS} \
  --position-embedding-type rope \
  --rotary-base ${ROPE_THETA} \
  --rotary-percent ${ROTARY_PERCENT} \
  --experimental-attention-variant gated_delta_net \
  --linear-attention-freq ${LINEAR_ATTN_FREQ} \
  --linear-conv-kernel-dim ${LINEAR_CONV_KERNEL_DIM} \
  --linear-key-head-dim ${LINEAR_KEY_HEAD_DIM} \
  --linear-value-head-dim ${LINEAR_VALUE_HEAD_DIM} \
  --linear-num-key-heads ${LINEAR_NUM_KEY_HEADS} \
  --linear-num-value-heads ${LINEAR_NUM_VALUE_HEADS} \
  --train-iters ${train_iters} \
  --lr 3e-5 \
  --lr-decay-iters ${train_iters} \
  --lr-decay-style cosine \
  --min-lr 3e-6 \
  --weight-decay 1e-2 \
  --clip-grad 1.0 \
  --lr-warmup-fraction .01 \
  ${PRECISION_ARGS} \
  ${OPTIMIZER_ARGS} \
  --use-flash-attn \
  --ckpt-format torch \
  --ddp-average-in-collective \
  --no-check-for-nan-in-loss-and-grad \
  --split 120,4,4 \
  --distributed-backend nccl \
  ${CHECKPOINT_ARGS} \
  ${DATASET_ARGS} \
  ${TOKENIZER_ARGS} \
  ${MOE_ARGS} \
  --log-interval 1 \
  --log-throughput \
  --save-interval 5000 \
  --eval-interval 5000 \
  --eval-iters 0 \
  --exit-interval 5000 \
  --transformer-impl transformer_engine \
  --cross-entropy-loss-fusion \
  --cross-entropy-fusion-impl native \
  --profile \
  --use-pytorch-profiler \
  --profile-step-start 3 \
  --profile-step-end 4 \
  --pytorch-profiler-collect-shapes \
  --tensorboard-dir ${OUTPUT_DIR}/logs \
"
# --pytorch-profiler-collect-callstack \


# run_cmd "${CMD}" "${LOG_DIR}" "${LOG_FILE}" "${node_rank}" "${node_num}"
mkdir -p "${LOG_DIR}"
echo "==== $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
echo "CMD: ${CMD}"
echo "LOG: ${LOG_FILE}"
set +e
bash -c "${CMD}" 2>&1 | tee "${LOG_FILE}"
_rc=${PIPESTATUS[0]}
set -e
if [[ "${_rc}" -ne 0 ]]; then
  echo "run failed rc=${_rc}; log tail:" >&2
  tail -n 60 "${LOG_FILE}" >&2 || true
  exit "${_rc}"
fi
