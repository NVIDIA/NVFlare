#!/usr/bin/env bash
workspace="workspace_brats"
n_clients=$1

if test -z "${n_clients}"
then
      echo "Usage: ./create_poc_workspace.sh [n_clients], e.g. ./create_poc_workspace.sh 8"
      exit 1
fi

# There should be $n_clients site-N folders.
python3 -m nvflare.lighter.poc -n "${n_clients}"

mv poc ${workspace}
# copy additional one for centralized training
cp -r ${workspace}/site-1 ${workspace}/site-All
