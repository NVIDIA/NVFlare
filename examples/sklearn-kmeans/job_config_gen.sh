#!/usr/bin/env bash

DATA_SPLIT_ROOT="/tmp/nvflare/iris_dataset"

prepare_job_config() {
    python3 utils/prepare_job_config.py --task_name "$1" --site_num "$2" \
    --split_method "$3" --data_split_root "$4"
}

echo "Generating job configs"
prepare_job_config sklearn_kmeans 3 uniform $DATA_SPLIT_ROOT
# prepare_job_config sklearn_kmeans 3 linear $DATA_SPLIT_ROOT
echo "Job configs generated"
