#!/usr/bin/env bash

# download dataset
python3 ../pt/utils/cifar10_download_data.py

RESULT_ROOT=/tmp/nvflare/sim_cifar10

# central
./set_alpha.sh cifar10_central 0.0
nvflare simulator job_configs/cifar10_central --workspace ${RESULT_ROOT}/central --threads 1 --n_clients 1

# FedAvg
./set_alpha.sh cifar10_fedavg 1.0
nvflare simulator job_configs/cifar10_fedavg --workspace ${RESULT_ROOT}/fedavg_alpha1.0 --threads 8 --n_clients 8
./set_alpha.sh cifar10_fedavg 0.5
nvflare simulator job_configs/cifar10_fedavg --workspace ${RESULT_ROOT}/fedavg_alpha0.5 --threads 8 --n_clients 8
./set_alpha.sh cifar10_fedavg 0.3
nvflare simulator job_configs/cifar10_fedavg --workspace ${RESULT_ROOT}/fedavg_alpha0.3 --threads 8 --n_clients 8
./set_alpha.sh cifar10_fedavg 0.1
nvflare simulator job_configs/cifar10_fedavg --workspace ${RESULT_ROOT}/fedavg_alpha0.1 --threads 8 --n_clients 8

# FedProx
./set_alpha.sh cifar10_fedprox 0.1
nvflare simulator job_configs/cifar10_fedprox --workspace ${RESULT_ROOT}/fedprox_alpha0.1 --threads 8 --n_clients 8

# FedOpt
./set_alpha.sh cifar10_fedopt 0.1
nvflare simulator job_configs/cifar10_fedopt --workspace ${RESULT_ROOT}/fedopt_alpha0.1 --threads 8 --n_clients 8

# SCAFFOLD
./set_alpha.sh cifar10_scaffold 0.1
nvflare simulator job_configs/cifar10_scaffold --workspace ${RESULT_ROOT}/scaffold_alpha0.1 --threads 8 --n_clients 8
