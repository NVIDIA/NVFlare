#!/usr/bin/env bash

prepare_job_config() {
    python3 utils/prepare_job_config.py --job_name "$1" --site_num "$2" --label_owner "$3"
}

echo "Generating job configs"
prepare_job_config vertical_xgb_psi 2 1
prepare_job_config vertical_xgb 2 1

echo "Job configs generated"
