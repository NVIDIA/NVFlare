#!/usr/bin/env bash

export PYTHONPATH=${PYTHONPATH}:${PWD}/..

# download dataset
./prepare_data.sh

# central
python ./jobs/cifar10_central/job.py --epochs 25

# FedAvg
python ./jobs/cifar10_fedavg/job.py --n_clients 8 --num_rounds 50 --alpha 1.0
python ./jobs/cifar10_fedavg/job.py --n_clients 8 --num_rounds 50 --alpha 0.5
python ./jobs/cifar10_fedavg/job.py --n_clients 8 --num_rounds 50 --alpha 0.3
python ./jobs/cifar10_fedavg/job.py --n_clients 8 --num_rounds 50 --alpha 0.1

# FedProx
python ./jobs/cifar10_fedprox/job.py --n_clients 8 --num_rounds 50 --alpha 0.1 --fedproxloss_mu 1e-5

# FedOpt
python ./jobs/cifar10_fedopt/job.py --n_clients 8 --num_rounds 50 --alpha 0.1

# SCAFFOLD
python ./jobs/cifar10_scaffold/job.py --n_clients 8 --num_rounds 50 --alpha 0.1
