#!/usr/bin/env bash
# add current folder to PYTHONPATH
export PYTHONPATH="${PWD}"
echo "PYTHONPATH is ${PYTHONPATH}"

algorithms_dir="${PWD}/configs"
servername="localhost"
workspace="workspaces/secure_workspace"
admin_username="admin@nvidia.com"
site_pre="site-"

n_clients=$1
config=$2
run=$3
alpha=$4

if test -z "${n_clients}" || test -z "${config}" || test -z "${run}" || test -z "${alpha}"
then
      echo "Usage: ./run_secure.sh [n_clients] [config] [run] [alpha], e.g. ./run_secure.sh 8 cifar10_fedavg 1 0.1"
      exit 1
fi

# start server
echo "STARTING SERVER"
export CUDA_VISIBLE_DEVICES=0
./${workspace}/${servername}/startup/start.sh &
sleep 10

# start clients
echo "STARTING ${n_clients} CLIENTS"
for id in $(eval echo "{1..$n_clients}")
do
  export CUDA_VISIBLE_DEVICES=0
  ./${workspace}/"${site_pre}${id}"/startup/start.sh &
done
sleep 10

# download and split data
echo "PREPARING DATA"
python3 ./pt/utils/prepare_data.py --data_dir="/tmp/cifar10_data" --num_sites="${n_clients}" --alpha="${alpha}"

# start training
echo "STARTING TRAINING"
python3 ./run_fl.py --port=8003 --admin_dir="./${workspace}/${admin_username}" \
  --username="${admin_username}" --run_number="${run}" --app="${algorithms_dir}/${config}" --min_clients="${n_clients}"

# sleep for FL system to shut down, so a new run can be started automatically
sleep 30
