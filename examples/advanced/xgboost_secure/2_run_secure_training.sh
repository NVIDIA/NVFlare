#!/bin/bash
set -e
world_size=2
python train_federated.py "${world_size}"
