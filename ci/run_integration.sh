#!/bin/bash
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
#   BUILD_TYPE: tests to execute, default = numpy

set -ex
BUILD_TYPE=numpy

if [[ $# -eq 1 ]]; then
    BUILD_TYPE=$1
elif [[ $# -gt 1 ]]; then
    echo "ERROR: too many parameters are provided"
    exit 1
fi

init_pipenv() {
    echo "initializing pip environment"
    rm -f Pipfile Pipfile.lock
    export PIPENV_INSTALL_TIMEOUT=9999
    export PIPENV_TIMEOUT=9999
    pipenv install -e .[dev]
    export PYTHONPATH=$PWD
}

remove_pipenv() {
    echo "removing pip environment"
    pipenv --rm
    rm Pipfile Pipfile.lock
}

integration_test_tf() {
    echo "Run TF integration test..."
    # since running directly in container, point python to python3.12
    ln -sfn /usr/bin/python3.12 /usr/bin/python
    ln -sfn /usr/bin/python3.12 /usr/bin/python3
    # somehow the base container has blinker which should be removed
    apt remove -y python3-blinker python-blinker-doc || true
    # pipenv does not work with TensorFlow so using pip
    python3.12 -m pip install -e .[dev]
    python3.12 -m pip install tensorflow[and-cuda]
    export PYTHONPATH=$PWD
    testFolder="tests/integration_test"
    clean_up_snapshot_and_job
    pushd ${testFolder}
    ./run_integration_tests.sh -m tensorflow
    popd
    clean_up_snapshot_and_job
}

add_dns_entries() {
    echo "adding DNS entries for HA test cases"
    cp /etc/hosts /etc/hosts_bak
    echo "127.0.0.1 localhost0 localhost1" | tee -a /etc/hosts > /dev/null
}

remove_dns_entries() {
    echo "restoring original /etc/hosts file"
    cp /etc/hosts_bak /etc/hosts
}

clean_up_snapshot_and_job() {
    rm -rf /tmp/nvflare*
}

integration_test() {
    echo "Run integration test with backend $1..."
    init_pipenv
    add_dns_entries
    testFolder="tests/integration_test"
    pushd ${testFolder}
    pipenv run ./generate_test_configs_for_examples.sh
    pipenv run ./run_integration_tests.sh -m "$1"
    popd
    clean_up_snapshot_and_job
    remove_dns_entries
    remove_pipenv
}

case $BUILD_TYPE in

    tensorflow)
        echo "Run TF tests..."
        integration_test_tf
        ;;
    *)
        integration_test "$BUILD_TYPE"
        ;;
esac
