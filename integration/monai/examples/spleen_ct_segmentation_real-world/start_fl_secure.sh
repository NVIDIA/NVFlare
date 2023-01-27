#!/usr/bin/env bash

# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

echo "PYTHONPATH is ${PYTHONPATH}"

servername="localhost"
workspace="${PWD}/workspaces/secure_workspace"
site_pre="site-"

n_clients=$1

if test -z "${n_clients}"
then
      echo "Usage: ./run_secure.sh [n_clients], e.g. ./run_secure.sh 2"
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
