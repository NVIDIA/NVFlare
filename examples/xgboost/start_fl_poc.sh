#!/usr/bin/env bash
# add current folder to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:${PWD}"
echo "PYTHONPATH is ${PYTHONPATH}"

# customized settings
n_clients=$1
workspace="${PWD}/workspaces/xgboost_workspace_${n_clients}"
site_pre="site-"

# run on localhost and default admin
servername="localhost"
admin_username="admin"

# get client IDs, a string, e.g. "All" or "I2CVB MSD NCI_ISBI_3T NCI_ISBI_Dx"
client_ids=$1

if test -z "${n_clients}"
then
      echo "Usage: ./start_fl_poc.sh [client_ids], e.g. ./start_fl_poc.sh 5"
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
for id in $(eval echo "{1..$n_clients}")
do
  gpu_idx=$((${i} % ${n_gpus}))
  echo "Starting ${site_pre}${id} on GPU ${gpu_idx}"
  export CUDA_VISIBLE_DEVICES=${gpu_idx}
  bash "${workspace}/${site_pre}${id}/startup/start.sh" "${servername}:8002:8003" "${site_pre}${id}" &
  i=$((i+1))
done
sleep 10
