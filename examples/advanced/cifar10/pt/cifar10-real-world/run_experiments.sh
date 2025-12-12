#!/usr/bin/env bash

# download dataset
./prepare_data.sh

# FedAvg with TensorBoard streaming
./submit_job.sh cifar10_fedavg_stream_tb 1.0

# FedAvg with HE
./submit_job.sh cifar10_fedavg_he 1.0
