#!/usr/bin/env bash

# NVFLARE INSTALL
NVFLARE_VERSION="2.3.0"
pip install 'nvflare>=${NVFLARE_VERSION}'

# set NVFLARE_HOME, which is used by nvflare poc to symlink examples
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export NVFLARE_HOME="$SCRIPT_DIR/../../"

# set NVFLARE_POC_WORKSPACE to the default workspace
NVFLARE_POC_WORKSPACE="/tmp/nvflare/poc"
export NVFLARE_POC_WORKSPACE

# prepare POC for 2 clients
echo "y" | nvflare poc prepare -n 2

if [ ! -L "${NVFLARE_POC_WORKSPACE}/example_project/prod_00/admin@nvidia.com/transfer" ]; then
     echo "'nvflare poc prepare' did not generate symlink '${NVFLARE_POC_WORKSPACE}/example_project/prod_00/admin@nvidia.com/transfer'"
     exit 1
fi

# start POC FL server and FL clients
nvflare poc start -ex admin@nvidia.com
