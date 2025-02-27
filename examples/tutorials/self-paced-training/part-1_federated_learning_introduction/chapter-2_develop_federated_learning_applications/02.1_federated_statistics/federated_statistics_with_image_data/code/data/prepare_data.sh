#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

INPUT_DATASET_PATH="/tmp/nvflare/image_stats/data"
OUTPUT_DATASET_PATH="/tmp/nvflare/image_stats/data"

python3 "${SCRIPT_DIR}"/utils/prepare_data.py --input_dir "${INPUT_DATASET_PATH}" --output_dir "${OUTPUT_DATASET_PATH}"