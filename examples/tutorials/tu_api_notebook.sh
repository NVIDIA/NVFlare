#!/usr/bin/env bash


SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

"${SCRIPT_DIR}"/tu_pre_start.sh

jupyter nbconvert --to notebook --inplace --execute "${SCRIPT_DIR}/flare_api.ipynb"

"${SCRIPT_DIR}"/tu_post_cleanup.sh