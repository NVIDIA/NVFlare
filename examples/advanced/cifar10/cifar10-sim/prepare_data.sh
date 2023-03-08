#!/bin/bash

script_dir="$( dirname -- "$0"; )";

python "${script_dir}"/../pt/utils/cifar10_data_utils.py
