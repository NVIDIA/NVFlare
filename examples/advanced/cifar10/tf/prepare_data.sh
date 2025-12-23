#!/usr/bin/env bash

# This script downloads CIFAR-10 dataset by running a simple Python command
python3 -c "from tensorflow.keras import datasets; datasets.cifar10.load_data(); print('CIFAR-10 dataset downloaded successfully')"
