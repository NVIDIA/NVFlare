#!/bin/bash

set -e

PYTHONPATH="${PWD}/../.."

# CRITICAL: Set gRPC environment variables before ANY imports that might use gRPC
# See: https://github.com/grpc/grpc/issues/28557
export GRPC_POLL_STRATEGY="poll"
export GRPC_ENABLE_FORK_SUPPORT="False"

backends=(numpy tensorflow pytorch auth preflight cifar stats xgboost client_api client_api_qa model_controller_api standalone)

usage()
{
    echo "Run integration tests of NVFlare."
    echo
    echo "Syntax: ./run_integration_tests.sh -m [-c] [-d]"
    echo "options:"
    echo "m     Which backend/test to run (options: ${backends[*]})."
    echo "c     Clean up integration test results."
    echo "d     Debug mode."
    echo
    exit 1
}

base_cmd="pytest -v --log-cli-level=INFO --capture=no"
no_args="true"
while getopts ":m:dc" option; do
    case "${option}" in
        m) # framework/backend
            m=${OPTARG}
            prefix="NVFLARE_TEST_FRAMEWORK=$m PYTHONPATH=${PYTHONPATH}"
            ;;
        d) # debug
            export FL_LOG_LEVEL=DEBUG
            base_cmd="pytest -vv --log-cli-level=DEBUG --capture=no"
            ;;
        c) # Clean up
            echo "Clean up integration tests result"
            rm -rf ./*_test.xml ./integration_test.xml
            ;;
        *) # Invalid option
            usage
            ;;
    esac
    no_args="false"
done
[[ "$no_args" == "true" ]] && usage
cmd="$base_cmd"
hosts_backup=""

has_localhost_aliases()
{
    python - <<'PY'
import socket
import sys

for host in ("localhost0", "localhost1"):
    try:
        addresses = {info[4][0] for info in socket.getaddrinfo(host, None)}
    except OSError:
        sys.exit(1)
    if "127.0.0.1" not in addresses:
        sys.exit(1)
PY
}

restore_localhost_aliases()
{
    if [[ -n "$hosts_backup" && -f "$hosts_backup" ]]; then
        echo "Restoring original /etc/hosts file."
        cp "$hosts_backup" /etc/hosts
        rm -f "$hosts_backup"
    fi
}

ensure_localhost_aliases()
{
    if has_localhost_aliases; then
        return
    fi

    if [[ ! -w /etc/hosts ]]; then
        echo "ERROR: localhost0 and localhost1 must resolve to 127.0.0.1 before running integration tests."
        echo "Run ci/run_integration.sh, or add this line to /etc/hosts before running this script directly:"
        echo "127.0.0.1 localhost0 localhost1"
        exit 1
    fi

    echo "Adding DNS entries for integration test localhost aliases."
    hosts_backup=$(mktemp)
    cp /etc/hosts "$hosts_backup"
    trap restore_localhost_aliases EXIT
    echo "127.0.0.1 localhost0 localhost1" >> /etc/hosts
}

run_preflight_check_test()
{
    echo "Running preflight check integration tests."
    ensure_localhost_aliases
    cmd="$cmd --junitxml=./integration_test.xml preflight_check_test.py"
    echo "$cmd"
    eval "$cmd"
}

run_system_test()
{
    echo "Running system integration tests with backend $m."
    cmd="$prefix $cmd --junitxml=./integration_test.xml system_test.py"
    echo "$cmd"
    eval "$cmd"
}

run_pytest_files()
{
    files=$(python -c "
import yaml, sys
with open('test_configs.yml') as f:
    cfg = yaml.safe_load(f)
for f in cfg.get('pytest_files', {}).get('$m', []):
    print(f)
")
    if [ -n "$files" ]; then
        for f in $files; do
            local xml_name="${f%.py}.xml"
            echo "Running standalone pytest file: $f"
            eval "$prefix $base_cmd --junitxml=./$xml_name $f"
        done
    fi
}

run_tensorflow()
{
    echo "Running integration tests using tensorflow related jobs."
    cmd="$prefix TF_FORCE_GPU_ALLOW_GROWTH=true $cmd --junitxml=./integration_test.xml system_test.py"
    python -c "import tensorflow; print('TF version is ' + tensorflow.__version__)"
    echo "$cmd"
    eval "$cmd"
}

has_system_tests=$(python -c "
import yaml
with open('test_configs.yml') as f:
    cfg = yaml.safe_load(f)
print('yes' if cfg.get('test_configs', {}).get('$m') else 'no')
")

if [[ $m == "tensorflow" ]]; then
    run_tensorflow
elif [[ $m == "preflight" ]]; then
    run_preflight_check_test
elif [[ $has_system_tests == "yes" ]]; then
    run_system_test
fi
run_pytest_files
