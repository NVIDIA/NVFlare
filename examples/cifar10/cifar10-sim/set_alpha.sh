#!/usr/bin/env bash
echo "PYTHONPATH is ${PYTHONPATH}"

algorithms_dir="${PWD}/job_configs"

config=$1
alpha=$2

echo "Submit secure job ${config} with alpha=${alpha}"
workspace="${PWD}/workspaces/secure_workspace"
admin_username="admin@nvidia.com"

# submit job
COMMAND="python3 ./set_alpha.py --job=${algorithms_dir}/${config} --alpha=${alpha}"
echo "Running: ${COMMAND}"
eval "${COMMAND}"
