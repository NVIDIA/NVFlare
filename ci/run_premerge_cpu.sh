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

set -ex

function cleanup_pip_deps() {
    echo "cleaning pip requirements"
    pip freeze > _temp_pip_requirements.txt
    pip uninstall -r _temp_pip_requirements.txt -y
    rm _temp_pip_requirements.txt
}

function install_pip_deps() {
    echo "installing pip requirements"
    pip install -r requirements-dev.txt
    export PYTHONPATH=$PWD
}

# Cleanup pip environment first
cleanup_pip_deps

## Unit Tests
install_pip_deps
./runtest.sh
cleanup_pip_deps


## Wheel Build
install_pip_deps
pip install build twine torch torchvision
python3 -m build --wheel
cleanup_pip_deps

