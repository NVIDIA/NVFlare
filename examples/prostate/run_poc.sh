#!/usr/bin/env bash
# add current folder to PYTHONPATH
export PYTHONPATH="${PWD}"
echo "PYTHONPATH is ${PYTHONPATH}"
 
# customized settings
algorithms_dir="${PWD}/configs"
workspace="workspace_prostate"
site_pre="client_"
admin_username="admin"

# run on localhost
servername="localhost"

# exp settings
config=$1 # choose the app to deploy from the folders in ${PWD}/configs/, e.g. prostate_fedavg
run=$2 # int number
client_ids=$3 # string, e.g. "I2CVB MSD NCI_ISBI_3T NCI_ISBI_Dx" 
if test -z "${config}" || test -z "${run}" || test -z "${client_ids}" 
then
      echo "Usage: ./run_poc.sh [config] [run] [client_ids]"
      exit 1
fi

# get num of gpus and assign to clients in a balanced manner
n_gpus=$(nvidia-smi --list-gpus | wc -l)
echo "There are ${n_gpus} GPUs."
# start server
echo "------- STARTING SERVER"
export CUDA_VISIBLE_DEVICES=
bash "${PWD}/${workspace}/server/startup/start.sh" ${servername} &
sleep 10
# start clients 
echo "------- STARTING ${n_clients} CLIENTS"
i=1
for id in ${client_ids}
do
  gpu_idx=$((${i} % ${n_gpus}))
  echo "Starting ${site_pre}${id} on GPU ${gpu_idx}"
  export CUDA_VISIBLE_DEVICES=${gpu_idx}
  bash "${PWD}/${workspace}/${site_pre}${id}/startup/start.sh" ${servername} "${site_pre}${id}" &
  i=$((i+1))
  sleep 10
done 
sleep 10

# start training
i=$((i-1))
echo "------- START TRAINING with min_clients ${i}"
python3 run_fl.py --port=8003 --admin_dir="${PWD}/${workspace}/${admin_username}" \
  --run_number="${run}" --app="${algorithms_dir}/${config}" --min_clients="${i}" --poc --host ${servername}

