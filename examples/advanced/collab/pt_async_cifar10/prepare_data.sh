#!/bin/bash

script_dir="$(cd "$(dirname -- "$0")" && pwd)"

PYTHONPATH="${script_dir}${PYTHONPATH:+:${PYTHONPATH}}" python3 -c \
  'from cifar10_data import download_cifar10; download_cifar10("/tmp/cifar10")'
