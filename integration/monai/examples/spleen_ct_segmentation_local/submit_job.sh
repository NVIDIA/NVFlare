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

algorithms_dir="${PWD}"

config=$1
poc=$2

if [ "${poc}" ]
then
  if [ "${poc}" != "--poc" ]; then
    echo "${poc} not supported to run POC mode. Use --poc"
  fi
  echo "Submit poc job: ${config}"
  workspace="/tmp/nvflare/poc"
  admin_username="admin"
else
  echo "Submit secure job: ${config}"
  workspace="${PWD}/workspaces/secure_workspace"
  admin_username="admin@nvidia.com"
fi

# submit job
COMMAND="python3 ./submit_job.py --admin_dir=${workspace}/${admin_username} --username=${admin_username} --job=${algorithms_dir}/${config} ${poc}"
echo "Running: ${COMMAND}"
eval "${COMMAND}"
