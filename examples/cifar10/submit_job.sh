#!/usr/bin/env bash
echo "PYTHONPATH is ${PYTHONPATH}"

algorithms_dir="${PWD}/job_configs"

config=$1
alpha=$2
poc=$3

if [ "${poc}" ]
then
  if [ "${poc}" != "--poc" ]; then
    echo "${poc} not supported to run POC mode. Use --poc"
  fi
  echo "Submit poc job ${config} with alpha=${alpha}"
  workspace="${PWD}/workspaces/poc_workspace"
  admin_username="admin"
else
  echo "Submit secure job ${config} with alpha=${alpha}"
  workspace="${PWD}/workspaces/secure_workspace"
  admin_username="admin@nvidia.com"
fi

# submit job
COMMAND="python3 ./submit_job.py --admin_dir=${workspace}/${admin_username} --username=${admin_username} --job=${algorithms_dir}/${config} --alpha=${alpha} ${poc}"
echo "Running: ${COMMAND}"
eval "${COMMAND}"
