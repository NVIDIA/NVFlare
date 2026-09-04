#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
DATASET_DIR="${DATA_DIR}/cifar-10-batches-py"
ARCHIVE="${DATA_DIR}/cifar-10-python.tar.gz"
URL="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

mkdir -p "${DATA_DIR}"

if [ -d "${DATASET_DIR}" ] && [ -f "${ARCHIVE}" ]; then
    echo "CIFAR-10 dataset already prepared."
    exit 0
fi

if [ ! -f "${ARCHIVE}" ]; then
    echo "Downloading CIFAR-10 dataset..."
    curl -L "${URL}" -o "${ARCHIVE}"
fi

if [ ! -d "${DATASET_DIR}" ]; then
    echo "Extracting CIFAR-10 dataset..."
    tar -xzf "${ARCHIVE}" -C "${DATA_DIR}"
fi

echo "CIFAR-10 dataset prepared successfully:"
echo "  Dataset: ${DATASET_DIR}"
echo "  Archive: ${ARCHIVE}"
