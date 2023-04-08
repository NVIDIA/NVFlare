#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DATASET_PATH="/tmp/nvflare/psi"

mkdir -p ${DATASET_PATH}
echo "copy ${SCRIPT_DIR}/data to ${DATASET_PATH} directory"
cp -r "${SCRIPT_DIR}"/data "${DATASET_PATH}"/.
