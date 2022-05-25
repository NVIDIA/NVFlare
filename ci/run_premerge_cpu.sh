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

function pip_uninstall_deps() {
    echo "pip uninstalling requirements: $1"
    pip uninstall -r $1 -y
}

function pip_install_deps() {
    echo "pip installing requirements: $1"
    pip install -r $1
    export PYTHONPATH=$PWD
}

## Unit Tests
pip_install_deps requirements-dev.txt
./runtest.sh
pip_uninstall_deps requirements-dev.txt


## Wheel Build
pip_install_deps requirements-dev.txt
pip install build twine torch torchvision
python3 -m build --wheel
pip_uninstall_deps requirements-dev.txt