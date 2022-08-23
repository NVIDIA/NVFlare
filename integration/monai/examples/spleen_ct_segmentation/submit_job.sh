#!/usr/bin/env bash
echo "PYTHONPATH is ${PYTHONPATH}"

algorithms_dir="${PWD}"

job_config=$1

echo "Submit poc job ${job_config}"
workspace="/tmp/nvflare/poc"
admin_username="admin"

# submit job
COMMAND="python3 ./submit_job.py --admin_dir=${workspace}/${admin_username} --username=${admin_username} --job=${algorithms_dir}/${job_config}"
echo "Running: ${COMMAND}"
eval "${COMMAND}"
