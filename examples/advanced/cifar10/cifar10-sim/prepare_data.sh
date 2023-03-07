#!/bin/bash

DATASET_PATH="/tmp/cifar10"

python -c "from torchvision.datasets import CIFAR10; CIFAR10(root='${DATASET_PATH}', download=True)"
