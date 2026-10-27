export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_BCCL_DUMP_ON_TIMEOUT=1
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_FILE=bccl_debug_%h_%p.log
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_TOPO_FILE="/usr/local/birensupa/all/latest/bccl/xml/topo_2c.xml"


START_DATE=$(date "+%Y_%m_%d_%H:%M:%S")
timestamp=$(date "+%Y_%m_%d_%H_%M_%S")
echo "START TIME: $START_DATE"

DAY_INFO=$(date "+%m_%d")

ML_OUTPUT_DIR=${ML_OUTPUT_DIR:-./output}
mkdir -p $ML_OUTPUT_DIR

EXPNAME=${EXPNAME:-experiment}
variant=${EXPNAME}_${DAY_INFO}_l${NLAYERS}_mb${MICRO_BATCH}_gb${GLOBAL_BATCH}_tp${TP_SIZE}_pp${PP_SIZE}_n${NNODES}

DATA_OUTPUT_PATH=$ML_OUTPUT_DIR/model_dir/chatglm2_${MODEL_SIZE}_zh
CHECKPOINT_PATH=$DATA_OUTPUT_PATH/checkpoints/$variant
REPO_PATH=$DATA_OUTPUT_PATH/experiment
TENSORBOARD_PATH=$REPO_PATH/tensorboard/$variant/${timestamp}
LOGS_PATH=$REPO_PATH/$variant/logs/$NODE_RANK
mkdir -p $LOGS_PATH

if [ ! -f "${DATA_PATH}.bin" ] || [ ! -f "${DATA_PATH}.idx" ]; then
  echo "Dataset files ${DATA_PATH}.bin / ${DATA_PATH}.idx not found" >&2
  exit 1
fi

_THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TOKENIZER_PATH=${TOKENIZER_PATH:-$_THIS_DIR/tokenizer}
if [ ! -d "$TOKENIZER_PATH" ]; then
  echo "$TOKENIZER_PATH not found"
  exit
fi
