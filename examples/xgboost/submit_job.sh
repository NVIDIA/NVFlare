#!/usr/bin/env bash
# add current folder to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:${PWD}"
echo "PYTHONPATH is ${PYTHONPATH}"

n_clients=$1
mode=$2
split=$3
lr=$4

algorithms_dir="${PWD}/job_configs"
workspace="${PWD}/workspaces/xgboost_workspace_${n_clients}"
config=

# default server and admin
admin_username="admin"

# get particular config
config="higgs_${n_clients}_${mode}_${split}_split_${lr}_lr"

# submit job
COMMAND="python3 ./submit_job.py --admin_dir=${workspace}/${admin_username} --username=${admin_username} --job=${algorithms_dir}/${config}"
echo "Running: ${COMMAND}"
eval "${COMMAND}"
