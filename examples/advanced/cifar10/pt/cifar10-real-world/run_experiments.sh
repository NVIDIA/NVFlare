#!/usr/bin/env bash

# download dataset
./prepare_data.sh

# FedAvg with MLFlow streaming
python cifar10_fedavg_mlflow/job.py --n_clients 8 --num_rounds 50 --alpha 1.0 --tracking_uri http://localhost:5000

# FedAvg with HE
python cifar10_fedavg_he/job.py --n_clients 8 --num_rounds 50 --alpha 1.0
