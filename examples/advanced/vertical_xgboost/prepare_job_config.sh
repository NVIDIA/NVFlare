#!/usr/bin/env bash

prepare_job_config() {
    python3 utils/prepare_job_config.py --site_num "$1" --label_owner "$2"
}

echo "Generating job configs"
prepare_job_config 2 1
prepare_job_config 5 1

echo "Job configs generated"
