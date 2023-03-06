#!/bin/bash

set -e

PYTHONPATH="${PWD}/../.."
backends=(numpy tensorflow pytorch overseer ha auth preflight cifar auto)

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

[ $# -eq 0 ] && usage
while getopts ":m:c:d" option; do
    case "${option}" in
        m) # framework/backend
            cmd="pytest --junitxml=./integration_test.xml -v --log-cli-level=INFO --capture=no"
            m=${OPTARG}
            if [[ " ${backends[*]} " =~ " ${m} " ]]; then
                # whatever you want to do when array contains value
                prefix="NVFLARE_TEST_FRAMEWORK=$m PYTHONPATH=${PYTHONPATH}"
            else
              usage
            fi
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
done

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
    echo "Running system integration tests."
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

if [[ $m == "numpy" ]]; then
    echo "Running integration tests using numpy related jobs."
    run_system_test
elif [[ $m == "tensorflow" ]]; then
    run_tensorflow
elif [[ $m == "pytorch" ]]; then
    echo "Running integration tests using pytorch related jobs."
    run_system_test
elif [[ $m == "ha" ]]; then
    echo "Running HA integration tests."
    run_system_test
elif [[ $m == "auth" ]]; then
    echo "Running federated authorization integration tests."
    run_system_test
elif [[ $m == "overseer" ]]; then
    run_overseer_test
elif [[ $m == "preflight" ]]; then
    run_preflight_check_test
elif [[ $m == "cifar" ]]; then
    echo "Running integration tests using cifar jobs."
    run_system_test
elif [[ $m == "auto" ]]; then
    echo "Running integration tests using auto generated jobs."
    run_system_test
fi
