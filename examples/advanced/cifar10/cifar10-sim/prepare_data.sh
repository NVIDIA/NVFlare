#!/bin/bash

DATA_ROOT="/tmp/cifar10"

python -c "from torchvision.datasets import CIFAR10; CIFAR10(root='${DATA_ROOT}', download=True)"
