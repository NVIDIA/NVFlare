#!/usr/bin/env bash

prepare_job_config() {
    python3 utils/prepare_job_config.py --site_num "$1" \
    --split_method "$2"
}

echo "Generating job configs"
prepare_job_config 5 exponential
prepare_job_config 5 square
prepare_job_config 5 uniform

prepare_job_config 20 exponential
prepare_job_config 20 square
prepare_job_config 20 uniform
echo "Job configs generated"
