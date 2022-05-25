#!/usr/bin/env bash
#
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

# Argument(s):
#   BUILD_TYPE:   all/specific_test_name, tests to execute

set -ex
BUILD_TYPE=all

if [[ $# -eq 1 ]]; then
    BUILD_TYPE=$1

elif [[ $# -gt 1 ]]; then
    echo "ERROR: too many parameters are provided"
    exit 1
fi

pip_uninstall_deps() {
    echo "pip uninstalling requirements: $1"
    pip uninstall -r $1 -y
}

pip_install_deps() {
    echo "pip installing requirements: $1"
    pip install -r $1
    export PYTHONPATH=$PWD
}

unit_test() {
    echo "Run unit test..."
    pip_install_deps requirements-dev.txt
    ./runtest.sh
    pip_uninstall_deps requirements-dev.txt
}

wheel_build() {
    echo "Run wheel build..."
    pip_install_deps requirements-dev.txt
    pip install build twine torch torchvision
    python3 -m build --wheel
    pip_uninstall_deps requirements-dev.txt
}

case $BUILD_TYPE in

    all)
        echo "Run all tests..."
        unit_test
        wheel_build
        ;;

    unit_test)
        unit_test
        ;;

    wheel_build )
        wheel_build
        ;;

    *)
        echo "ERROR: unknown parameter: $BUILD_TYPE"
        ;;
esac