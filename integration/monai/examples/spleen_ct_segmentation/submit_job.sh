#!/usr/bin/env bash
echo "PYTHONPATH is ${PYTHONPATH}"

algorithms_dir="${PWD}"

config=$1

echo "Submit poc job ${config}"
workspace="/tmp/nvflare/poc"
admin_username="admin"

# submit job
COMMAND="python3 ./submit_job.py --admin_dir=${workspace}/${admin_username} --username=${admin_username} --job=${algorithms_dir}/${config}"
echo "Running: ${COMMAND}"
eval "${COMMAND}"
