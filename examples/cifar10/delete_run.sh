#!/usr/bin/env bash

workspace="workspaces/$1"
run_number=$2
servername="localhost"
site_pre="site-"

n_clients=8

if test -z "$1" || test -z "${run_number}"
then
      echo "Usage: ./delete_run.sh [workspace] [run_number], e.g. ./delete_run.sh poc_workspace 1"
      exit 1
fi

# delete server run
rm -r "./${workspace}/${servername}/run_${run_number}"

# delete client runs
for id in $(eval echo "{1..$n_clients}")
do
  rm -r "./${workspace}/${site_pre}${id}/run_${run_number}"
done
