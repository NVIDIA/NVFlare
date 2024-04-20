#!/usr/bin/env bash

export PYTHONPATH=${PYTHONPATH}:${PWD}/..

# download dataset
./prepare_data.sh

# RESULT_ROOT=/tmp/nvflare/sim_cifar10 is set in run_simulator.sh

# fedbn
./run_simulator.sh cifar10_fedbn 1.0 8 8
# ./run_simulator.sh cifar10_fedbn 0.5 8 8
# ./run_simulator.sh cifar10_fedbn 0.3 8 8
# ./run_simulator.sh cifar10_fedbn 0.1 8 8
