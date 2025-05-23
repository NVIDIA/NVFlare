#!/usr/bin/env bash
DATASET_PATH="DATASET_ROOT/HIGGS.csv"
WORKSPACE_ROOT="/tmp/nvflare/random_forest/HIGGS/workspaces"

n=5
for subsample in 0.5 0.05 0.005
do
    for study in uniform_split_uniform_lr exponential_split_uniform_lr exponential_split_scaled_lr 
    do
        echo ${n}_clients_${study}_split_${subsample}_subsample
        python3 utils/model_validation.py --data_path $DATASET_PATH --model_path $WORKSPACE_ROOT/workspace_${n}_${subsample}_${study}/server/simulate_job/app_server/xgboost_model.json --size_valid 1000000 --num_trees 100
    done
done


n=20
for subsample in 0.8 0.2 0.02
do
    for study in uniform_split_uniform_lr square_split_uniform_lr square_split_scaled_lr 
    do
        echo ${n}_clients_${study}_split_${subsample}_subsample
        python3 utils/model_validation.py --data_path $DATASET_PATH --model_path $WORKSPACE_ROOT/workspace_${n}_${subsample}_${study}/server/simulate_job/app_server/xgboost_model.json --size_valid 1000000 --num_trees 100
    done
done
