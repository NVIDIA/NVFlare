#!/usr/bin/env bash

# download dataset
python3 ../pt/utils/cifar10_download_data.py

# FedAvg with TensorBoard streaming
./submit_job.sh cifar10_fedavg_stream_tb 1.0

# FedAvg with HE
./submit_job.sh cifar10_fedavg_he 1.0
