#!/usr/bin/env bash
echo "PYTHONPATH is ${PYTHONPATH}"

algorithms_dir="${PWD}/jobs"

job=$1
alpha=$2
poc=$3

if [ "${poc}" ]
then
  if [ "${poc}" != "--poc" ]; then
    echo "${poc} not supported to run POC mode. Use --poc"
  fi
  echo "Submit poc job ${job} with alpha=${alpha}"
  workspace="/tmp/nvflare/poc"
  admin_username="admin"
else
  echo "Submit secure job ${job} with alpha=${alpha}"
  workspace="${PWD}/workspaces/secure_workspace"
  admin_username="admin@nvidia.com"
fi

# submit job
COMMAND="python3 ./submit_job.py --admin_dir=${workspace}/${admin_username} --username=${admin_username} --job=${algorithms_dir}/${job} --alpha=${alpha} ${poc}"
echo "Running: ${COMMAND}"
eval "${COMMAND}"
