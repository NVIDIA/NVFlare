#!/usr/bin/env bash
echo "PYTHONPATH is ${PYTHONPATH}"

algorithms_dir="${PWD}/configs"
workspace="${PWD}/workspace_brats"
site_pre="site-"

# default server and admin
servername="server"
admin_username="admin"

# get particular config
config=$1

# submit job
COMMAND="python3 ./submit_job.py --admin_dir=${workspace}/${admin_username} --username=${admin_username} --job=${algorithms_dir}/${config}"
echo "Running: ${COMMAND}"
eval "${COMMAND}"