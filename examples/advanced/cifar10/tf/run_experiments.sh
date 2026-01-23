#!/usr/bin/env bash

# Source TensorFlow environment variables
source ./set_tf_env_vars.sh

# download dataset
./prepare_data.sh

# central
python cifar10_central/train.py --epochs 25

# FedAvg
python cifar10_fedavg/job.py --n_clients 8 --num_rounds 50 --alpha 1.0
python cifar10_fedavg/job.py --n_clients 8 --num_rounds 50 --alpha 0.5
python cifar10_fedavg/job.py --n_clients 8 --num_rounds 50 --alpha 0.3
python cifar10_fedavg/job.py --n_clients 8 --num_rounds 50 --alpha 0.1

# FedProx
python cifar10_fedprox/job.py --n_clients 8 --num_rounds 50 --alpha 0.1 --fedprox_mu 1e-5

# FedOpt
python cifar10_fedopt/job.py --n_clients 8 --num_rounds 50 --alpha 0.1

# SCAFFOLD
python cifar10_scaffold/job.py --n_clients 8 --num_rounds 50 --alpha 0.1
