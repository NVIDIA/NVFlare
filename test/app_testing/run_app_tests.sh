#!/usr/bin/env bash

pip install tensorflow torch torchvision
python -c "from torchvision.datasets import CIFAR10; CIFAR10(root='~/data', download=True)"

TF_FORCE_GPU_ALLOW_GROWTH=true python test_runner.py --poc ../../nvflare/poc --n_clients 2 --yaml test_simple.yml --app_path test_apps --cleanup
