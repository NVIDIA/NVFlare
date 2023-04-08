#!/usr/bin/env bash
algorithms_dir="${PWD}/jobs"

job=$1
alpha=$2

echo "Configure job ${job} using alpha=${alpha}"
workspace="${PWD}/workspaces/secure_workspace"
admin_username="admin@nvidia.com"

# set alpha
COMMAND="python3 ./set_alpha.py --job=${algorithms_dir}/${job} --alpha=${alpha}"
echo "Running: ${COMMAND}"
eval "${COMMAND}"
