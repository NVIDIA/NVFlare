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
TEST_FOLDER="tests/integration_test"
PYTHON_BIN=(python)
PYTEST_ARGS=(-v --log-cli-level=INFO --capture=no)
XGBOOST_FEDERATED_WHEEL_URL="https://s3-us-west-2.amazonaws.com/xgboost-nightly-builds/federated-secure/xgboost-2.2.0.dev0%2B4601688195708f7c31fcceeb0e0ac735e7311e61-py3-none-manylinux_2_28_x86_64.whl"

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
    local status
    # since running directly in container, point python to python3.12
    ln -sfn /usr/bin/python3.12 /usr/bin/python
    ln -sfn /usr/bin/python3.12 /usr/bin/python3
    # somehow the base container has blinker which should be removed
    apt remove -y python3-blinker python-blinker-doc || true
    # pipenv does not work with TensorFlow so using pip
    python3.12 -m pip install -e .[dev]
    python3.12 -m pip install tensorflow[and-cuda]
    export PYTHONPATH=$PWD
    PYTHON_BIN=(python3.12)
    clean_up_snapshot_and_job
    pushd ${TEST_FOLDER}
    set +e
    run_pytest_mode tensorflow
    status=$?
    set -e
    popd
    clean_up_snapshot_and_job
    return $status
}

add_dns_entries() {
    echo "adding DNS entries for integration test cases"
    cp /etc/hosts /etc/hosts_bak
    echo "127.0.0.1 localhost0" | tee -a /etc/hosts > /dev/null
}

remove_dns_entries() {
    echo "restoring original /etc/hosts file"
    cp /etc/hosts_bak /etc/hosts
}

clean_up_snapshot_and_job() {
    rm -rf /tmp/nvflare*
}

run_pytest() {
    "${PYTHON_BIN[@]}" -m pytest "${PYTEST_ARGS[@]}" --junitxml=./integration_test.xml "$@"
}

install_xgboost_federated_wheel() {
    echo "Installing federated XGBoost wheel for XGBoost recipe tests..."
    "${PYTHON_BIN[@]}" -m pip install --force-reinstall --no-deps "${XGBOOST_FEDERATED_WHEEL_URL}"
    "${PYTHON_BIN[@]}" - <<'PY'
import xgboost
import xgboost.federated

print("XGBoost version is " + xgboost.__version__)
print("xgboost.federated import succeeded")
PY
}

run_system_test() {
    local test_mode=$1
    local status
    export NVFLARE_TEST_FRAMEWORK=$test_mode
    run_pytest system_test.py
    status=$?
    unset NVFLARE_TEST_FRAMEWORK
    return $status
}

run_tensorflow_test() {
    local status
    "${PYTHON_BIN[@]}" -c "import tensorflow; print('TF version is ' + tensorflow.__version__)"
    status=$?
    if [[ $status -ne 0 ]]; then
        return $status
    fi
    export NVFLARE_TEST_FRAMEWORK=tensorflow
    export TF_FORCE_GPU_ALLOW_GROWTH=true
    run_pytest system_test.py
    status=$?
    unset TF_FORCE_GPU_ALLOW_GROWTH
    unset NVFLARE_TEST_FRAMEWORK
    return $status
}

run_pytest_mode() {
    local test_mode=$1
    case $test_mode in
        fast)
            run_pytest fast
            ;;
        slow)
            install_xgboost_federated_wheel
            run_pytest slow
            ;;
        auto)
            echo "The legacy auto example discovery mode is currently disabled; no tests to run."
            ;;
        tensorflow)
            run_tensorflow_test
            ;;
        numpy|pytorch|auth|cifar|stats|xgboost|client_api|client_api_qa|model_controller_api)
            run_system_test "$test_mode"
            ;;
        *)
            echo "ERROR: unknown integration test mode: $test_mode"
            exit 1
            ;;
    esac
}

integration_test() {
    echo "Run integration test mode $1..."
    local status
    init_pipenv
    add_dns_entries
    PYTHON_BIN=(pipenv run python)
    pushd ${TEST_FOLDER}
    set +e
    run_pytest_mode "$1"
    status=$?
    set -e
    popd
    clean_up_snapshot_and_job
    remove_dns_entries
    remove_pipenv
    return $status
}

integration_test_pt() {
    echo "Run PT integration test mode $1..."
    local status
    ln -sfn /usr/bin/python3.12 /usr/bin/python
    ln -sfn /usr/bin/python3.12 /usr/bin/python3
    # somehow the base container has blinker which should be removed
    apt remove -y python3-blinker python-blinker-doc || true
    pip install -e .[dev]
    # CI machine supports CUDA 12.4; pin to known compatible versions
    pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
    export PYTHONPATH=$PWD
    add_dns_entries
    PYTHON_BIN=(python)
    clean_up_snapshot_and_job
    pushd ${TEST_FOLDER}
    set +e
    run_pytest_mode "$1"
    status=$?
    set -e
    popd
    clean_up_snapshot_and_job
    remove_dns_entries
    return $status
}

case $BUILD_TYPE in

    tensorflow)
        echo "Run TF tests..."
        integration_test_tf
        ;;
    client_api|client_api_qa|pytorch|cifar)
        echo "Run PT tests..."
        integration_test_pt "$BUILD_TYPE"
        ;;
    *)
        integration_test "$BUILD_TYPE"
        ;;
esac
