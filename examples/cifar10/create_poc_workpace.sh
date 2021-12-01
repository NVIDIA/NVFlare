#!/usr/bin/env bash
workspace="poc_workspace"

n_clients=$1

if test -z "${n_clients}"
then
      echo "Usage: ./create_poc_workspace.sh [n_clients], e.g. ./create_poc_workspace.sh 8"
      exit 1
fi

cur_dir=${PWD}

# create POC startup kits
cd "workspaces" || exit
python3 -m nvflare.lighter.poc -n "${n_clients}" || exit
# There should be $n_clients site-N folders.

# move the folder
mv "poc" ${workspace}
echo "Created POC workspace at ./workspaces/${workspace}"

cd "${cur_dir}" || exit
