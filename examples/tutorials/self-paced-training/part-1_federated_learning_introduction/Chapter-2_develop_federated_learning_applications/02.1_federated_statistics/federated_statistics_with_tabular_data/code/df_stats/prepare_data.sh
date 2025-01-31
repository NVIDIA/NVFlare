#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DATASET_PATH="/tmp/nvflare/df_stats/data"
python3 "${SCRIPT_DIR}"/utils/prepare_data.py  --prepare-data --dest "${DATASET_PATH}"