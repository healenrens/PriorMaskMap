#!/usr/bin/env bash
CONFIG=$1
GPUS=$2
shift 2
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd):${PYTHONPATH}"
python -m torch.distributed.launch --nproc_per_node="$GPUS" \
  --master_port="${PORT:-29531}" tools/train.py "$CONFIG" --launcher pytorch "$@"
