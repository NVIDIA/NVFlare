#!/usr/bin/env bash

# central
./run_secure.sh 1 cifar10_central 1 1.0

# FedAvg
./run_secure.sh 8 cifar10_fedavg 2 1.0
./run_secure.sh 8 cifar10_fedavg 3 0.5
./run_secure.sh 8 cifar10_fedavg 4 0.3
./run_secure.sh 8 cifar10_fedavg 5 0.1

# FedProx
./run_secure.sh 8 cifar10_fedprox 6 0.1

# FedOpt
./run_secure.sh 8 cifar10_fedopt 7 0.1

# FedAvg + HE
./run_secure.sh 8 cifar10_fedavg_he 8 1.0
