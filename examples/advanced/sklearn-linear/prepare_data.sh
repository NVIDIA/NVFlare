#!/bin/bash

DATASET_DIR="/tmp/nvflare/dataset"
DATASET_PATH="${DATASET_DIR}/HIGGS.csv"

if [ -f "$DATASET_PATH" ]; then
    echo "${DATASET_PATH} exists."
else
    mkdir -p "${DATASET_DIR}"
    # Please note that the UCI's website may experience occasional downtime.
    # overwrite the file if it exists so that corrupted downloads don't persist
    wget -O "${DATASET_DIR}/higgs.zip" https://archive.ics.uci.edu/static/public/280/higgs.zip
    unzip -o "${DATASET_DIR}/higgs.zip" -d "${DATASET_DIR}"
    gzip -d -f "${DATASET_DIR}/HIGGS.csv.gz"
    rm -f "${DATASET_DIR}/higgs.zip"

    echo "Data downloaded and saved in ${DATASET_PATH}"
fi
