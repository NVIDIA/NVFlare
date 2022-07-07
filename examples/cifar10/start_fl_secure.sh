#!/usr/bin/env bash
# add current folder to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:${PWD}"
echo "PYTHONPATH is ${PYTHONPATH}"

servername="localhost"
workspace="${PWD}/workspaces/secure_workspace"
site_pre="site-"

n_clients=$1

if test -z "${n_clients}"
then
      echo "Usage: ./run_secure.sh [n_clients], e.g. ./run_secure.sh 8"
      exit 1
fi

# start server
echo "STARTING SERVER"
export CUDA_VISIBLE_DEVICES=0  # in case FedOpt uses GPU
"${workspace}/${servername}/startup/start.sh" &
sleep 10

# start clients
echo "STARTING ${n_clients} CLIENTS"
for id in $(eval echo "{1..$n_clients}")
do
  #export CUDA_VISIBLE_DEVICES=0  # Client GPU resources will be managed by nvflare
  "${workspace}/${site_pre}${id}/startup/start.sh" &
done
sleep 10
