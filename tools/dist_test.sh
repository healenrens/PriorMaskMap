#!/usr/bin/env bash
CONFIG=$1
CHECKPOINT=$2
GPUS=$3
shift 3
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd):${PYTHONPATH}"
python -m torch.distributed.launch --nproc_per_node="$GPUS" \
  --master_port="${PORT:-29541}" tools/test.py "$CONFIG" "$CHECKPOINT" --launcher pytorch "$@"
