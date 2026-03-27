#!/usr/bin/env bash
set -euo pipefail

DATASET_PATH="${1}/HIGGS.csv"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [ ! -f "${DATASET_PATH}" ]
then
    if [ "${NVFLARE_XGB_ALLOW_SYNTHETIC_DATA:-false}" = "true" ]
    then
        echo "HIGGS dataset not found at ${DATASET_PATH}; generating synthetic fallback dataset."
        mkdir -p "$(dirname "${DATASET_PATH}")"
        python3 - "${DATASET_PATH}" <<'PY'
import csv
import random
import sys

out_path = sys.argv[1]
rows = 20000
cols = 29  # 1 label + 28 features
random.seed(42)

with open(out_path, "w", newline="") as f:
    w = csv.writer(f)
    for _ in range(rows):
        label = random.randint(0, 1)
        feats = [f"{random.random():.6f}" for _ in range(cols - 1)]
        w.writerow([label, *feats])
PY
    else
        echo "Please check if you saved HIGGS dataset in ${DATASET_PATH}"
        exit 1
    fi
fi

echo "Generating HIGGS data splits, reading from ${DATASET_PATH}"

DATA_ROWS=$(wc -l < "${DATASET_PATH}")
if [ "${DATA_ROWS}" -lt 10 ]
then
    echo "Dataset at ${DATASET_PATH} has too few rows (${DATA_ROWS})"
    exit 1
fi

SIZE_TOTAL=${DATA_ROWS}
SIZE_VALID=$(( DATA_ROWS / 10 ))
if [ "${SIZE_VALID}" -lt 1 ]; then SIZE_VALID=1; fi
if [ "${SIZE_VALID}" -ge "${SIZE_TOTAL}" ]; then SIZE_VALID=$(( SIZE_TOTAL / 2 )); fi

echo "Using size_total=${SIZE_TOTAL}, size_valid=${SIZE_VALID} based on dataset rows=${DATA_ROWS}"

OUTPUT_PATH="/tmp/nvflare/dataset/xgboost_higgs_horizontal"
for site_num in 2 5 20;
do
    for split_mode in uniform exponential square;
    do
        python3 "${SCRIPT_DIR}/utils/prepare_data_horizontal.py" \
        --data_path "${DATASET_PATH}" \
        --site_num ${site_num} \
        --size_total ${SIZE_TOTAL} \
        --size_valid ${SIZE_VALID} \
        --split_method ${split_mode} \
        --out_path "${OUTPUT_PATH}/${site_num}_${split_mode}"
    done
done
echo "Horizontal data splits are generated in ${OUTPUT_PATH}"

OUTPUT_PATH="/tmp/nvflare/dataset/xgboost_higgs_vertical"
OUTPUT_FILE="higgs.data.csv"
# Note: HIGGS has 11 million preshuffled instances; using rows_total_percentage to reduce PSI time for example
python3 "${SCRIPT_DIR}/utils/prepare_data_vertical.py" \
--data_path "${DATASET_PATH}" \
--site_num 2 \
--rows_total_percentage 0.02 \
--rows_overlap_percentage 0.25 \
--out_path "${OUTPUT_PATH}" \
--out_file "${OUTPUT_FILE}"
echo "Vertical data splits are generated in ${OUTPUT_PATH}"
