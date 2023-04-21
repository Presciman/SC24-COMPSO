#!/bin/bash
#BSUB -nnodes 1
#BSUB -W 02:00
#BSUB -P csc337
#BSUB -alloc_flags "smt4 nvme"
#BSUB -J error
#BSUB -o %J.output_train_eigen.txt
#BSUB -e %J.err_train_eigen.txt
#BSUB -q batch

cd /ccs/home/dtao/baixi/BERT-PyTorch/requirements/kfac-pytorch/examples
# Load modules
module load open-ce/1.5.2-py38-0
module load gcc
conda activate bert-pytorch

nprocspn=6
nnodes=$(cat ${LSB_DJOB_HOSTFILE} | sort | uniq | grep -v login | grep -v batch | wc -l)
nprocs=$(( ${nnodes} * ${nprocspn} ))

echo Number of nodes ${nnodes}
echo nprocspn ${nprocspn}
#Baixi: important for enabling 6 gpus per compute node
unset CUDA_VISIBLE_DEVICES

jsrun --smpiargs="-disable_gpu_hooks" -n ${nnodes} -g 6 -c 42 -a ${nprocspn} -r 1 -E OMP_NUM_THREADS=8 --bind=proportional-packed:7 --launch_distribution=packed ./launch.sh python3 torch_cifar10_resnet.py --data-dir /gpfs/alpine/csc337/scratch/dtao/baixi/error_impact/cifar10 --log-dir /gpfs/alpine/csc337/scratch/dtao/baixi/error_impact/logs_eigen --error_bound 1e-1



