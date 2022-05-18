#!/usr/bin/env bash
# add current folder to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:${PWD}"
echo "PYTHONPATH is ${PYTHONPATH}"

# customized settings
algorithms_dir="${PWD}/configs"
workspace="${PWD}/workspace_brats"
site_pre="site-"

# run on localhost and default admin
servername="localhost"
admin_username="admin"

# get client IDs, a string, e.g. "All" or "1 2"
client_ids=$1

if test -z "${client_ids}"
then
      echo "Usage: ./start_fl_poc.sh [client_ids], e.g. ./start_fl_poc.sh \"All\""
      exit 1
fi

# get num of gpus and assign to clients in a balanced manner
n_gpus=$(nvidia-smi --list-gpus | wc -l)
echo "There are ${n_gpus} GPUs."
# start server
echo "STARTING SERVER"
export CUDA_VISIBLE_DEVICES=
bash "${workspace}/server/startup/start.sh" ${servername} &
sleep 10

# start clients
echo "STARTING CLIENTS"
i=0
for id in ${client_ids}
do
  gpu_idx=$((${i} % ${n_gpus}))
  echo "Starting ${site_pre}${id} on GPU ${gpu_idx}"
  export CUDA_VISIBLE_DEVICES=${gpu_idx}
  bash "${workspace}/${site_pre}${id}/startup/start.sh" "${servername}:8002:8003" "${site_pre}${id}" &
  i=$((i+1))
done
sleep 10
