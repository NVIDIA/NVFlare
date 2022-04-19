#!/usr/bin/env bash

pip install tensorflow torch torchvision

TF_FORCE_GPU_ALLOW_GROWTH=true pytest -v test_system.py -s
