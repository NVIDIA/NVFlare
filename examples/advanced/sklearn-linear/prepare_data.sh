#!/bin/bash

DATASET_DIR="/tmp/nvflare/dataset"
DATASET_PATH="${DATASET_DIR}/HIGGS.csv"

if [ -f "$DATASET_PATH" ]; then
    echo "${DATASET_PATH} exists."
else
    wget --directory-prefix ${DATASET_DIR} https://archive.ics.uci.edu/static/public/280/higgs.zip
    unzip "${DATASET_DIR}/higgs.zip"
    gzip -d "${DATASET_DIR}/HIGGS.csv.gz"

    echo "Data downloaded and saved in ${DATASET_PATH}"
fi
