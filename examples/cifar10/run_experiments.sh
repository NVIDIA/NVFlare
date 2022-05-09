#!/usr/bin/env bash

# central
./submit_job.sh cifar10_central 0.0

# FedAvg
./submit_job.sh cifar10_fedavg 1.0
./submit_job.sh cifar10_fedavg 0.5
./submit_job.sh cifar10_fedavg 0.3
./submit_job.sh cifar10_fedavg 0.1

# FedProx
./submit_job.sh cifar10_fedprox 0.1

# FedOpt
./submit_job.sh cifar10_fedopt 0.1

# SCAFFOLD
./submit_job.sh cifar10_scaffold 0.1

# FedAvg + HE
./submit_job.sh cifar10_fedavg_he 1.0

# FedAvg with TensorBoard streaming
./submit_job.sh cifar10_fedavg_stream_tb 1.0
