#!/usr/bin/env bash

TREE_METHOD="hist"
DATA_SPLIT_ROOT="/tmp/nvflare/random_forest/HIGGS/data_splits"

prepare_job_config() {
    python3 utils/prepare_job_config.py --site_num "$1" --num_local_parallel_tree "$2" --local_subsample "$3" \
    --split_method "$4" --lr_mode "$5" --nthread 16 --tree_method "$6" --data_split_root "$7"
}

echo "Generating job configs"
prepare_job_config 5 20 0.5 exponential scaled $TREE_METHOD $DATA_SPLIT_ROOT
prepare_job_config 5 20 0.5 exponential uniform $TREE_METHOD $DATA_SPLIT_ROOT
prepare_job_config 5 20 0.5 uniform uniform $TREE_METHOD $DATA_SPLIT_ROOT
prepare_job_config 5 20 0.05 exponential scaled $TREE_METHOD $DATA_SPLIT_ROOT
prepare_job_config 5 20 0.05 exponential uniform $TREE_METHOD $DATA_SPLIT_ROOT
prepare_job_config 5 20 0.05 uniform uniform $TREE_METHOD $DATA_SPLIT_ROOT
prepare_job_config 5 20 0.005 exponential scaled $TREE_METHOD $DATA_SPLIT_ROOT
prepare_job_config 5 20 0.005 exponential uniform $TREE_METHOD $DATA_SPLIT_ROOT
prepare_job_config 5 20 0.005 uniform uniform $TREE_METHOD $DATA_SPLIT_ROOT

prepare_job_config 20 5 0.8 square scaled $TREE_METHOD $DATA_SPLIT_ROOT
prepare_job_config 20 5 0.8 square uniform $TREE_METHOD $DATA_SPLIT_ROOT
prepare_job_config 20 5 0.8 uniform uniform $TREE_METHOD $DATA_SPLIT_ROOT
prepare_job_config 20 5 0.2 square scaled $TREE_METHOD $DATA_SPLIT_ROOT
prepare_job_config 20 5 0.2 square uniform $TREE_METHOD $DATA_SPLIT_ROOT
prepare_job_config 20 5 0.2 uniform uniform $TREE_METHOD $DATA_SPLIT_ROOT
prepare_job_config 20 5 0.02 square scaled $TREE_METHOD $DATA_SPLIT_ROOT
prepare_job_config 20 5 0.02 square uniform $TREE_METHOD $DATA_SPLIT_ROOT
prepare_job_config 20 5 0.02 uniform uniform $TREE_METHOD $DATA_SPLIT_ROOT

echo "Job configs generated"
