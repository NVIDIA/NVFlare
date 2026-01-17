#!/usr/bin/env bash

# This script downloads CIFAR-10 dataset by running a simple Python command
# First, clean up any existing/corrupted dataset files to ensure a fresh download
echo "Cleaning up existing CIFAR-10 cache..."
rm -rf ~/.keras/datasets/cifar-10-batches-py*
echo "✓ Cache cleared"
echo ""
echo "Downloading CIFAR-10 dataset..."
python3 -c "from tensorflow.keras import datasets; datasets.cifar10.load_data(); print('✓ CIFAR-10 dataset downloaded successfully')"
