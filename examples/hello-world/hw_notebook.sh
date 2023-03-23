#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
"${SCRIPT_DIR}"/hw_pre_start.sh

# wait for system to start
sleep 30

jupyter nbconvert --to notebook --inplace --execute "${SCRIPT_DIR}"/hello_world.ipynb

"${SCRIPT_DIR}"/hw_post_cleanup.sh





