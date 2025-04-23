#!/bin/bash

python3 -c "from torchvision.datasets import CIFAR10; CIFAR10(root='/tmp/nvflare/data/cifar10', download=True)"
