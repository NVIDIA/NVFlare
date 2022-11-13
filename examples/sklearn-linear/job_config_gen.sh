#!/usr/bin/env bash

DATA_SPLIT_ROOT="${PWD}/data_splits"

prepare_job_config() {
    python3 utils/prepare_job_config.py --site_num "$1" \
    --split_method "$2" --data_split_root "$3"
}

echo "Generating job configs"
prepare_job_config 5 exponential $DATA_SPLIT_ROOT
prepare_job_config 5 square $DATA_SPLIT_ROOT
prepare_job_config 5 uniform $DATA_SPLIT_ROOT

prepare_job_config 20 exponential $DATA_SPLIT_ROOT
prepare_job_config 20 square $DATA_SPLIT_ROOT
prepare_job_config 20 uniform $DATA_SPLIT_ROOT

echo "Job configs generated"
