#!/usr/bin/env bash

set -e

pip install tensorflow==2.8.0 torch torchvision

TF_FORCE_GPU_ALLOW_GROWTH=true pytest --junitxml=./integration_test.xml -v system_test.py overseer_test.py -s
