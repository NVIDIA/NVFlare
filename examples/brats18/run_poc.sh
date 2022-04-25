#!/usr/bin/env bash
# add current folder to PYTHONPATH
export PYTHONPATH="${PWD}"
echo "PYTHONPATH is ${PYTHONPATH}"

algorithms_dir="${PWD}/configs"
servername="localhost"
workspace="workspaces/poc_workspace"
admin_username="admin"  # default admin
site_pre="site-"


n_clients=$1 # int number
config=$2 # choose from the folders in ${PWD}/configs
run=$3 # int number
dataset_base_dir=$4 #e.g. "/workspace/dataset/brats18/"
datalist_json_path=$5 #e.g. "${PWD}/datalists/brats/brats_13clients"


if test -z "${n_clients}" || test -z "${config}" || test -z "${run}" || test -z "${dataset_base_dir}"
then
      echo "Usage: ./run_poc.sh [n_clients] [config] [run] [dataset_base_dir], e.g. ./run_poc.sh 2 brats_fedavg_dp 1 /workspace/dataset/brats18/"
      exit 1
fi

n_gpus=$(nvidia-smi --list-gpus | wc -l)
echo "There are ${n_gpus} GPUs."

# start server
echo "------- STARTING SERVER"
export CUDA_VISIBLE_DEVICES=0
./${workspace}/server/startup/start.sh ${servername} &
# We need IP address of the server when clients and server are runing on different machine
echo "The server starts on IP address: $(hostname --all-ip-addresses)" 
sleep 10

# start clients on different cuda device
echo "------- STARTING ${n_clients} CLIENTS"
for id in $(eval echo "{1..$n_clients}")
do
  gpu_idx=$((${id} % ${n_gpus}))
  echo "Starting client${id} on GPU ${gpu_idx}"
  export CUDA_VISIBLE_DEVICES=${gpu_idx}
  bash "./${workspace}/${site_pre}${id}/startup/start.sh" ${servername} "${site_pre}${id}" &
  sleep 10
done
sleep 10

# generate training config data
echo "------- PREPARING TRAINING CONFIG"
echo python3 ./pt/utils/prepare_train_config.py --site_pre ${site_pre} --num_sites ${n_clients} -o "${algorithms_dir}/${config}/config" --dataset_base_dir=${dataset_base_dir} --datalist_json_path="${datalist_json_path}/config_brats18_datalist_client0.json"
python3 ./pt/utils/prepare_train_config.py --site_pre ${site_pre} --num_sites ${n_clients} -o "${algorithms_dir}/${config}/config" --dataset_base_dir=${dataset_base_dir} --datalist_json_path="${datalist_json_path}/config_brats18_datalist_client0.json"

# start training
echo "------- START TRAINING"
python3 ./run_fl.py --port=8003 --admin_dir="./${workspace}/${admin_username}" \
  --run_number="${run}" --app="${algorithms_dir}/${config}" --min_clients="${n_clients}" --poc --host ${servername}

# # sleep for FL system to shut down, so a new run can be started automatically
# sleep 30
