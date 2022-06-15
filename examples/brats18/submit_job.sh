#!/usr/bin/env bash
export PYTHONPATH="${PYTHONPATH}:${PWD}"
echo "PYTHONPATH is ${PYTHONPATH}"

algorithms_dir="${PWD}/configs"
workspace="${PWD}/workspace_brats"

# default server and admin
servername="server"
admin_username="admin"

# get particular config
config=$1

# submit job
COMMAND="python3 ./submit_job.py --admin_dir=${workspace}/${admin_username} --username=${admin_username} --job=${algorithms_dir}/${config}"
echo "Running: ${COMMAND}"
eval "${COMMAND}"