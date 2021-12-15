#!/usr/bin/env bash
dataset_base_dir='/dataset/brats18/'
datalist_json_path='${PWD}/datalists/brats/brats_13clients'
datalist_json_path_central='${PWD}/datalists/brats/brats_1clients'

# central
./run_poc.sh 1 brats18_central 1 $dataset_base_dir $datalist_json_path_central

# FedAvg
./run_poc.sh 13 brats18_fedavg 2 $dataset_base_dir $datalist_json_path

# FedAvg with DP
./run_poc.sh 13 brats18_fedavg_dp 3 $dataset_base_dir $datalist_json_path
