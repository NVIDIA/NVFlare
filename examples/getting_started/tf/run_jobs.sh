#!/usr/bin/env bash

export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_ALLOCATOR=cuda_malloc_asyncp


# You can change GPU index if multiple GPUs are available
GPU_INDX=0

# You can change workspace - where results and artefact will be saved.
WORKSPACE=/tmp

# Run centralized training job
python ./tf_fl_script_executor_cifar10.py \
       --algo centralized \
       --n_clients 1 \
       --num_rounds 25 \
       --batch_size 64 \
       --epochs 1 \
       --alpha 0.0 \
       --gpu $GPU_INDX \
       --workspace $WORKSPACE


# Run FedAvg with different alpha values
for alpha in 1.0 0.5 0.3 0.1; do

    python ./tf_fl_script_executor_cifar10.py \
       --algo fedavg \
       --n_clients 8 \
       --num_rounds 50 \
       --batch_size 64 \
       --epochs 4 \
       --alpha $alpha \
       --gpu $GPU_INDX \
       --workspace $WORKSPACE

done


# Run FedOpt job
python ./tf_fl_script_executor_cifar10.py \
       --algo fedopt \
       --n_clients 8 \
       --num_rounds 50 \
       --batch_size 64 \
       --epochs 4 \
       --alpha 0.1 \
       --gpu $GPU_INDX \
       --workspace $WORKSPACE
