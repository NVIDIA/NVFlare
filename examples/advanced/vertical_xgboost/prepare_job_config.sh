#!/usr/bin/env bash

prepare_job_config() {
    python3 utils/prepare_job_config.py --site_num "$1"
}

echo "Generating job configs"
prepare_job_config 2
prepare_job_config 5

echo "Job configs generated"
