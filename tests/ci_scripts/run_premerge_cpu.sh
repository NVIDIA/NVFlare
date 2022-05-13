#!/bin/bash
#
# Copyright (c) 2020-2022, NVIDIA CORPORATION. All rights reserved.
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
#

set -ex


## Unit Tests
# output formatting
separator=""
blue=""
green=""
red=""
noColor=""

if [[ -t 1 ]] # stdout is a terminal
then
    separator=$'--------------------------------------------------------------------------------\n'
    blue="$(tput bold; tput setaf 4)"
    green="$(tput bold; tput setaf 2)"
    red="$(tput bold; tput setaf 1)"
    noColor="$(tput sgr0)"
fi

function print_style_fail_msg() { 
    echo "${red}Check failed!${noColor}" 
    echo "Please run ${green}./runtest.sh${noColor} to check errors and fix them." 
} 

set +e
folders_to_check_license="nvflare tests"

grep -r --include "*.py" --exclude-dir "*protos*" -L "\(# Copyright (c) \(2021\|2021-2022\|2022\), NVIDIA CORPORATION.  All rights reserved.\)\|\(This file is released into the public domain.\)" ${folders_to_check_license} > no_license.lst
if [ -s no_license.lst ]; then
    # The file is not-empty.
    cat no_license.lst
    echo "License text not found on the above files."
    echo "Please fix them."
    rm -f no_license.lst
    exit 1
else
    echo "All Python files in folder (${folders_to_check_license}) have license header"
    rm -f no_license.lst
fi

set -e
export PYTHONPATH=$(pwd):$PYTHONPATH && echo "PYTHONPATH is ${PYTHONPATH}"

python3 -m flake8 nvflare

echo "${separator}${blue}isort-fix${noColor}"
python3 -m isort --check $PWD/nvflare
isort_status=$?
if [ ${isort_status} -ne 0 ]
then
    print_style_fail_msg
else
    echo "${green}passed!${noColor}"
fi

echo "${separator}${blue}black-fix${noColor}"
python3 -m black --check $PWD/nvflare 
black_status=$?
if [ ${black_status} -ne 0 ]
then
    print_style_fail_msg
else
    echo "${green}passed!${noColor}"
fi
echo "Done with isort/black code style checks"

set +e
echo "${separator}${blue}pydocstyle-fix${noColor}"
python3 -m pydocstyle $PWD/nvflare 
black_status=$?
if [ ${black_status} -ne 0 ]
then
    echo "docstring check failed"
else
    echo "${green}passed!${noColor}"
fi
echo "Done with pydocstyle docstring style checks"

echo "Running unit tests"
wget "https://dl.min.io/server/minio/release/linux-amd64/minio"
chmod +x ./minio
export MINIO_STORE_PATH=nvflare_unittest_minio_server
export MINIO_ROOT_USER=admin
export MINIO_ROOT_PASSWORD=password
export MINIO_SERVER_PORT=9001
./minio server $MINIO_STORE_PATH --console-address :$MINIO_SERVER_PORT &
echo $! > ./minio_pid
minio_pid=`cat ./minio_pid`
python -m pytest --cov=nvflare --cov-report html:cov_html --cov-report xml:cov.xml --numprocesses=auto tests/unit_test/
kill -9 $minio_pid
rm -rf ./$MINIO_STORE_PATH ./minio_pid ./minio
echo "Done with unit tests"



## Wheel Build
pip install -r $PWD/../../requirements-dev.txt
pip install build twine torch torchvision
python3 -m build --wheel
