#!/bin/bash

set -e

PYTHONPATH="${PWD}/../.."
backends=(numpy tensorflow pytorch overseer auth preflight cifar auto stats xgboost client_api client_api_qa model_controller_api)

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

no_args="true"
while getopts ":m:dc" option; do
    case "${option}" in
        m) # framework/backend
            cmd="pytest --junitxml=./integration_test.xml -v --log-cli-level=INFO --capture=no"
            m=${OPTARG}
            prefix="NVFLARE_TEST_FRAMEWORK=$m PYTHONPATH=${PYTHONPATH}"
            ;;
        d) # debug
            export FL_LOG_LEVEL=DEBUG
            cmd="pytest --junitxml=./integration_test.xml -vv --log-cli-level=DEBUG --capture=no"
            ;;
        c) # Clean up
            echo "Clean up integration tests result"
            rm -rf ./integration_test.xml
            ;;
        *) # Invalid option
            usage
            ;;
    esac
    no_args="false"
done
[[ "$no_args" == "true" ]] && usage

run_preflight_check_test()
{
    echo "Running preflight check integration tests."
    cmd="$cmd preflight_check_test.py"
    echo "$cmd"
    eval "$cmd"
}

run_overseer_test()
{
    echo "Running overseer integration tests."
    cmd="$cmd overseer_test.py"
    echo "$cmd"
    eval "$cmd"
}

run_system_test()
{
    echo "Running system integration tests with backend $m."
    cmd="$prefix $cmd system_test.py"
    echo "$cmd"
    eval "$cmd"
}

run_tensorflow()
{
    echo "Running integration tests using tensorflow related jobs."
    cmd="$prefix TF_FORCE_GPU_ALLOW_GROWTH=true $cmd system_test.py"
    python -c "import tensorflow; print('TF version is ' + tensorflow.__version__)"
    echo "$cmd"
    eval "$cmd"
}

if [[ $m == "tensorflow" ]]; then
    run_tensorflow
elif [[ $m == "overseer" ]]; then
    run_overseer_test
# elif [[ $m == "preflight" ]]; then
#     run_preflight_check_test
else
    run_system_test
fi
