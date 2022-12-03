#!/usr/bin/env bash

prepare_job_config() {
    python3 utils/prepare_job_config.py --site_num "$1" \
    --split_method "$2"
}

echo "Generating job configs"
prepare_job_config 3 uniform
prepare_job_config 3 linear
echo "Job configs generated"
