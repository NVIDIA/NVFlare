#!/bin/bash

script_dir="$( dirname -- "$0"; )";
cifar10_src="${script_dir}"/../../cifar10/pt/src

PYTHONPATH="${cifar10_src}${PYTHONPATH:+:${PYTHONPATH}}" python3 "${cifar10_src}"/data/cifar10_data_utils.py
