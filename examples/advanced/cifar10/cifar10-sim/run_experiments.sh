#!/usr/bin/env bash

export PYTHONPATH=${PYTHONPATH}:${PWD}/..

# download dataset
./prepare_data.sh

# RESULT_ROOT=/tmp/nvflare/sim_cifar10 is set in run_simulator.sh

# central
./run_simulator.sh cifar10_central 0.0 1 1

# FedAvg
./run_simulator.sh cifar10_fedavg 1.0 8 8
./run_simulator.sh cifar10_fedavg 0.5 8 8
./run_simulator.sh cifar10_fedavg 0.3 8 8
./run_simulator.sh cifar10_fedavg 0.1 8 8

# FedProx
./run_simulator.sh cifar10_fedprox 0.1 8 8

# FedOpt
./run_simulator.sh cifar10_fedopt 0.1 8 8

# SCAFFOLD
./run_simulator.sh cifar10_scaffold 0.1 8 8
