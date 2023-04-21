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

CMD+="torch_cifar10_resnet.py --data-dir ./cifar10 --log-dir ./logs_$S_MODE"

if [[ "$S_MODE" -eq 0 ]]; then
    CMD+="--use-inv-kfac "
elif [[ "$S_MODE" -eq 2 ]]; then
    CMD+="--error_bound 1e-4 "
fi

echo "Training Command: $CMD"
$CMD
