#!/bin/bash
set -e
mkdir -p ./model
world_size=2
python train_federated.py "${world_size}"
