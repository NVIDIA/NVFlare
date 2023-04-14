#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
echo "$SCRIPT_DIR"
"${SCRIPT_DIR}"/hw_pre_start.sh

jupyter nbconvert --to notebook --inplace --execute "${SCRIPT_DIR}/hello_world.ipynb"

#wait for system to stop
sleep 10

"${SCRIPT_DIR}"/hw_post_cleanup.sh

echo "finished"
