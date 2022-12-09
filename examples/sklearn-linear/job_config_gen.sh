#!/usr/bin/env bash

DATA_SPLIT_ROOT="/tmp/nvflare/higgs_dataset"

prepare_job_config() {
    python3 utils/prepare_job_config.py --task_name "$1" --site_num "$2" \
    --split_method "$3" --data_split_root "$4"
}

echo "Generating job configs"
prepare_job_config sklearn_linear 5 uniform $DATA_SPLIT_ROOT
# prepare_job_config sklearn_linear 5 exponential $DATA_SPLIT_ROOT
# prepare_job_config sklearn_linear 5 square $DATA_SPLIT_ROOT
#prepare_job_config sklearn_linear 20 uniform $DATA_SPLIT_ROOT
#prepare_job_config sklearn_linear 20 exponential $DATA_SPLIT_ROOT
#prepare_job_config sklearn_linear 20 square $DATA_SPLIT_ROOT
echo "Job configs generated"
