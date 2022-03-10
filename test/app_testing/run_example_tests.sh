#!/usr/bin/env bash

pip install tensorflow

python test_runner.py --poc ../../nvflare/poc --n_clients 2 --yaml test_examples.yml --app_path ../../examples --cleanup
