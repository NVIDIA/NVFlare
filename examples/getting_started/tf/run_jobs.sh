#!/usr/bin/env bash

export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_ALLOCATOR=cuda_malloc_asyncp


# You can change GPU index if multiple GPUs are available
GPU_INDX=0


# Run centralized training job
python ./tf_fl_script_executor_cifar10.py \
       --algo centralized \
       --n_clients 1 \
       --num_rounds 1 \
       --batch_size 64 \
       --epochs 25 \
       --alpha 0.0 \
       --gpu $GPU_INDX

# Run FedAvg with different alpha values
for alpha in 1.0 0.5 0.3 0.1; do

    python ./tf_fl_script_executor_cifar10.py \
       --algo fedavg \
       --n_clients 8 \
       --num_rounds 50 \
       --batch_size 64 \
       --epochs 4 \
       --alpha $alpha \
       --gpu $GPU_INDX

done


# Run FedProx job.
python ./tf_fl_script_executor_cifar10.py \
       --algo fedprox \
       --n_clients 8 \
       --num_rounds 50 \
       --batch_size 64 \
       --epochs 4 \
       --fedprox_mu 1e-5 \
       --alpha 0.1 \
       --gpu $GPU_INDX


# Run FedOpt job
python ./tf_fl_script_executor_cifar10.py \
       --algo fedopt \
       --n_clients 8 \
       --num_rounds 50 \
       --batch_size 64 \
       --epochs 4 \
       --alpha 0.1 \
       --gpu $GPU_INDX
