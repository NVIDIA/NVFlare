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

grep -V
echo "set to +e"
set +e
grep -r --include "*.py" -L "# Copyright (c) 2021, NVIDIA CORPORATION." nvflare > no_license.lst
if [ -s no_license.lst ]; then
    # The file is not-empty.
    cat no_license.lst
    echo "License text not found on the above files."
    echo "Please fix them."
    rm -f no_license.lst
    exit 1
else
    echo "All Python files in nvflare have license header"
    rm -f no_license.lst
fi

export PWD=$(pwd) && echo $PWD
export PYTHONPATH=$PWD:$PYTHONPATH && echo $PYTHONPATH

set -e

# now only flake8
python3 -m flake8 nvflare

# set +e  # disable exit on failure so that diagnostics can be given on failure
echo "${separator}${blue}isort-fix${noColor}"
python3 -m isort --check $PWD/nvflare
isort_status=$?
if [ ${isort_status} -ne 0 ]
then
    print_style_fail_msg
else
    echo "${green}passed!${noColor}"
fi
# set -e # enable exit on failure

# set +e  # disable exit on failure so that diagnostics can be given on failure
echo "${separator}${blue}black-fix${noColor}"
python3 -m black --check $PWD/nvflare 
black_status=$?
if [ ${black_status} -ne 0 ]
then
    print_style_fail_msg
else
    echo "${green}passed!${noColor}"
fi
# set -e
echo "Done with flake tests"

echo "Running unit tests"
pytest --numprocesses=auto test
echo "Done with unit tests"
