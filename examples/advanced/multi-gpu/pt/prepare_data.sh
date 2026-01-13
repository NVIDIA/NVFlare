#!/bin/bash

DATASET_ROOT="/tmp/nvflare/data"

echo "Downloading CIFAR-10 dataset to ${DATASET_ROOT}..."
python3 -c "import torchvision.datasets as datasets; datasets.CIFAR10(root='${DATASET_ROOT}', train=True, download=True)"

echo "Dataset downloaded successfully!"

