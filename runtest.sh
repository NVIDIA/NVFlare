#!/usr/bin/env bash 

# output formatting
separator=""
blue=""
green=""
red=""
noColor=""

if [[ -t 1 ]] # stdout is a terminal
then
    separator=$'--------------------------------------------------------------------------------\n'
    blue="$(tput bold; tput setaf 4)"
    green="$(tput bold; tput setaf 2)"
    red="$(tput bold; tput setaf 1)"
    noColor="$(tput sgr0)"
fi

function print_style_fail_msg() { 
    echo "${red}Check failed!${noColor}" 
    echo "Please run ${green}./runtest.sh${noColor} to check errors and fix them." 
} 

set +e
folders_to_check_license="nvflare test"

grep -r --include "*.py" -L "\(# Copyright (c) \(2021\|2021-2022\|2022\), NVIDIA CORPORATION.  All rights reserved.\)\|\(This file is released into the public domain.\)" ${folders_to_check_license} > no_license.lst
if [ -s no_license.lst ]; then
    # The file is not-empty.
    cat no_license.lst
    echo "License text not found on the above files."
    echo "Please fix them."
    rm -f no_license.lst
    exit 1
else
    echo "All Python files in folder (${folders_to_check_license}) have license header"
    rm -f no_license.lst
fi

set -e
export PYTHONPATH=$(pwd):$PYTHONPATH && echo "PYTHONPATH is ${PYTHONPATH}"

python3 -m flake8 nvflare

echo "${separator}${blue}isort-fix${noColor}"
python3 -m isort --check $PWD/nvflare
isort_status=$?
if [ ${isort_status} -ne 0 ]
then
    print_style_fail_msg
else
    echo "${green}passed!${noColor}"
fi

echo "${separator}${blue}black-fix${noColor}"
python3 -m black --check $PWD/nvflare 
black_status=$?
if [ ${black_status} -ne 0 ]
then
    print_style_fail_msg
else
    echo "${green}passed!${noColor}"
fi
echo "Done with isort/black code style checks"

set +e
echo "${separator}${blue}pydocstyle-fix${noColor}"
python3 -m pydocstyle $PWD/nvflare 
black_status=$?
if [ ${black_status} -ne 0 ]
then
    echo "docstring check failed"
else
    echo "${green}passed!${noColor}"
fi
echo "Done with pydocstyle docstring style checks"

set -e
echo "Running unit tests"
pytest --numprocesses=auto test
echo "Done with unit tests"
