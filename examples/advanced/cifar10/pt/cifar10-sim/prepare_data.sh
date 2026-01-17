#!/bin/bash

script_dir="$( dirname -- "$0"; )";

python3 "${script_dir}"/../src/data/cifar10_data_utils.py
