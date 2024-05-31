#!/usr/bin/env bash

export TF_FORCE_GPU_ALLOW_GROWTH=true 
export TF_GPU_ALLOCATOR=cuda_malloc_asyncp

python tf_fedavg_script_executor_cifar10_alpha_split.py --n_clients 8 --alpha 1.0 --gpu 0
python tf_fedavg_script_executor_cifar10_alpha_split.py --n_clients 8 --alpha 0.5 --gpu 1
python tf_fedavg_script_executor_cifar10_alpha_split.py --n_clients 8 --alpha 0.3 --gpu 2
python tf_fedavg_script_executor_cifar10_alpha_split.py --n_clients 8 --alpha 0.1 --gpu 4


#export CUDA_VISIBLE_DEVICES=0; 
nvflare simulator -w ./workdir/cifar10_tf_fedavg_alpha1.0 -n 8 -t 8 /tmp/nvflare/jobs/job_config/cifar10_tf_fedavg_alpha1.0 &
#export CUDA_VISIBLE_DEVICES=1; nvflare simulator -w ./workdir/cifar10_tf_fedavg_alpha0.5 -n 8 -t 8 /tmp/nvflare/jobs/job_config/cifar10_tf_fedavg_alpha0.5 &
#export CUDA_VISIBLE_DEVICES=2; nvflare simulator -w ./workdir/cifar10_tf_fedavg_alpha0.3 -n 8 -t 8 /tmp/nvflare/jobs/job_config/cifar10_tf_fedavg_alpha0.3 &
#export CUDA_VISIBLE_DEVICES=3; nvflare simulator -w ./workdir/cifar10_tf_fedavg_alpha0.1 -n 8 -t 8 /tmp/nvflare/jobs/job_config/cifar10_tf_fedavg_alpha0.1
