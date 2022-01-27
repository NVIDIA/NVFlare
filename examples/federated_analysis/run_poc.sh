#!/usr/bin/env bash
# add current folder to PYTHONPATH
export PYTHONPATH="${PWD}"
echo "PYTHONPATH is ${PYTHONPATH}"

algorithms_dir="${PWD}/configs"
servername="localhost"
workspace="workspaces/poc_workspace"
admin_username="admin"  # default admin
site_pre="site-"

n_clients=$1
config=$2
run=$3

if test -z "${n_clients}" || test -z "${config}" || test -z "${run}"
then
      echo "Usage: ./run_poc.sh [n_clients] [config] [run], e.g. ./run_poc.sh 4 fed_analysis 1"
      exit 1
fi

# start server
echo "STARTING SERVER"
export CUDA_VISIBLE_DEVICES=0
./${workspace}/server/startup/start.sh ${servername} &
sleep 10

# start clients
echo "STARTING ${n_clients} CLIENTS"
for id in $(eval echo "{1..$n_clients}")
do
  export CUDA_VISIBLE_DEVICES=0
  ./${workspace}/"${site_pre}${id}"/startup/start.sh ${servername} "${site_pre}${id}" &
done
sleep 10

# start training
echo "STARTING ANALYSIS"
python3 ./run_fl.py --port=8003 --admin_dir="./${workspace}/${admin_username}" \
  --run_number="${run}" --app="${algorithms_dir}/${config}" --min_clients="${n_clients}" --poc

# sleep for FL system to shut down, so a new run can be started automatically
sleep 30
