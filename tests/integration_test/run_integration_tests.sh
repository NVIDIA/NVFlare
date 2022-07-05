#!/bin/bash

set -e

usage()
{
    echo "Run integration tests of NVFlare."
    echo
    echo "Syntax: ./run_integration_tests.sh -m [-c]"
    echo "options:"
    echo "m     Which framework/backend to test (options: numpy, tensorflow, pytorch)."
    echo "c     Clean up integration test results."
    echo
    exit 1
}

cmd="pytest --junitxml=./integration_test.xml -v -s"
[ $# -eq 0 ] && usage
while getopts ":m:c" option; do
    case "${option}" in
        m) # framework/backend
            m=${OPTARG}
            [[ $m == "numpy" || $m == "tensorflow" || $m == "pytorch" ]] || usage
            prefix="NVFLARE_TEST_FRAMEWORK=$m"
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

run_numpy()
{
    echo "Running integration tests using numpy related jobs."
    cmd="$prefix $cmd system_test.py overseer_test.py"
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

run_pytorch()
{
    echo "Running integration tests using pytorch related jobs."
    cmd="$prefix $cmd system_test.py"
    python -m pip install tensorboard torch torchvision
    python -c "import torch; print('PyTorch version is ' + torch.__version__)"
    echo "$cmd"
    eval "$cmd"
}

if [[ $m == "numpy" ]]; then
    run_numpy
elif [[ $m == "tensorflow" ]]; then
    run_tensorflow
elif [[ $m == "pytorch" ]]; then
    run_pytorch
fi
