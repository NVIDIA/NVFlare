#!/usr/bin/env bash

# T2a system with 1 client
export CUDA_VISIBLE_DEVICES=0
./workspaces/t2a_workspace/localhost/startup/start.sh
./workspaces/t2a_workspace/site_a-1/startup/start.sh
