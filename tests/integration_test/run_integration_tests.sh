#!/usr/bin/env bash

set -e

pip install tensorflow torch torchvision

TF_FORCE_GPU_ALLOW_GROWTH=true pytest --junitxml=./integration_test.xml -v system_test.py -s
