#!/bin/bash

python -c "from torchvision.datasets import CIFAR10; CIFAR10(root='/tmp/nvflare/tensorboard-streaming', download=True)"
