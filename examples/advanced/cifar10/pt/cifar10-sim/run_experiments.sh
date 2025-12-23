#!/usr/bin/env bash

export PYTHONPATH=${PYTHONPATH}:${PWD}/..

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
python cifar10_fedprox/job.py --n_clients 8 --num_rounds 50 --alpha 0.1 --fedproxloss_mu 1e-5

# FedOpt
python cifar10_fedopt/job.py --n_clients 8 --num_rounds 50 --alpha 0.1

# SCAFFOLD
python cifar10_scaffold/job.py --n_clients 8 --num_rounds 50 --alpha 0.1

# Custom Aggregators
python cifar10_custom_aggr/job.py --aggregator weighted --n_clients 8 --num_rounds 50 --alpha 0.1
python cifar10_custom_aggr/job.py --aggregator median --n_clients 8 --num_rounds 50 --alpha 0.1
python cifar10_custom_aggr/job.py --aggregator default --n_clients 8 --num_rounds 50 --alpha 0.1
