#!/usr/bin/env bash
# add current folder to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:${PWD}"
echo "PYTHONPATH is ${PYTHONPATH}"

servername="localhost"
workspace="${PWD}/workspaces/poc_workspace"
site_pre="site-"

n_clients=$1

if test -z "${n_clients}"
then
      echo "Usage: ./run_poc.sh [n_clients], e.g. ./run_poc.sh 8"
      exit 1
fi

# start server
echo "STARTING SERVER"
"${workspace}/server/startup/start.sh" ${servername} &
sleep 10

# start clients
echo "STARTING ${n_clients} CLIENTS"
for id in $(eval echo "{1..$n_clients}")
do
  "${workspace}/${site_pre}${id}/startup/start.sh" "${servername}:8002:8003" "${site_pre}${id}" &
done
sleep 10
