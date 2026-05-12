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
PYTEST_ARGS=(-v --log-cli-level=INFO --capture=no)

# CRITICAL: Set gRPC environment variables before ANY imports that might use gRPC.
# See: https://github.com/grpc/grpc/issues/28557
export GRPC_POLL_STRATEGY="poll"
export GRPC_ENABLE_FORK_SUPPORT="False"

if [[ $# -eq 1 ]]; then
    BUILD_TYPE=$1

elif [[ $# -gt 1 ]]; then
    echo "ERROR: too many parameters are provided"
    exit 1
fi

init_pipenv() {
    echo "initializing pip environment"
    pipenv install -e .[dev]
    export PYTHONPATH=$PWD
}

remove_pipenv() {
    echo "removing pip environment"
    pipenv --rm
    rm Pipfile Pipfile.lock
}

add_dns_entries() {
    echo "adding DNS entries for integration test cases"
    cp /etc/hosts /etc/hosts_bak
    echo "127.0.0.1 localhost0 localhost1" | tee -a /etc/hosts > /dev/null
}

remove_dns_entries() {
    echo "restoring original /etc/hosts file"
    cp /etc/hosts_bak /etc/hosts
}

integration_test() {
    echo "Run integration test..."
    local status
    add_dns_entries
    testFolder="tests/integration_test"
    rm -rf /tmp/nvflare*
    pushd ${testFolder}
    set +e
    NVFLARE_TEST_FRAMEWORK=numpy pipenv run python -m pytest "${PYTEST_ARGS[@]}" --junitxml=./integration_test.xml system_test.py
    status=$?
    if [[ $status -eq 0 ]]; then
        pipenv run python -m pytest "${PYTEST_ARGS[@]}" --junitxml=./fast_integration_test.xml fast
        status=$?
    fi
    set -e
    popd
    rm -rf /tmp/nvflare*
    remove_dns_entries
    return $status
}

unit_test() {
    echo "Run unit test..."
    pipenv run ./runtest.sh
}

wheel_build() {
    echo "Run wheel build..."
    pipenv install build twine
    pipenv run python -m build --wheel
}


case $BUILD_TYPE in

    all)
        echo "Run all tests..."
        init_pipenv
        unit_test
        integration_test
        wheel_build
        remove_pipenv
        ;;

    unit_test)
        init_pipenv
        unit_test
        remove_pipenv
        ;;

    integration_test)
        init_pipenv
        integration_test
        remove_pipenv
        ;;

    wheel_build )
        init_pipenv
        wheel_build
        remove_pipenv
        ;;

    *)
        echo "ERROR: unknown parameter: $BUILD_TYPE"
        ;;
esac
