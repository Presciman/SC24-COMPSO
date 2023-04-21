#!/bin/bash

NNODES=1
HOST="localhost"
NGPUS=4
CMD="python3 -m torch.distributed.run "
CMD+="--nnodes=$NNODES --nproc_per_node=$NGPUS --max_restarts 0 "
if [[ "$NNODES" -eq 1 ]]; then
    CMD+="--standalone "
else
    CMD+="--rdzv_backend=c10d --rdzv_endpoint=$HOST "
fi

mkdir ./logs_$S_MODE

CMD+="bert_comm.py"

echo "Command: $CMD"
$CMD

