#!/usr/bin/env bash
n_clients=$1

if test -z "${n_clients}"
then
      echo "Usage: ./create_poc_workspace.sh [n_clients], e.g. ./create_poc_workspace.sh 8"
      exit 1
fi

cur_dir=${PWD}
workspace="xgboost_workspace_${n_clients}"

# create POC startup kits
mkdir "workspaces"
cd "workspaces" || exit
python3 -m nvflare.lighter.poc -n "${n_clients}" || exit
# There should be $n_clients site-N folders.

# move the folder
mv "poc" "${workspace}"
echo "Created POC workspace at ./workspaces/${workspace}"

cd "${cur_dir}" || exit
