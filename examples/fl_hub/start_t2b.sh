#!/usr/bin/env bash

# T2b system with 2 clients
export CUDA_VISIBLE_DEVICES=1
./workspaces/t2b_workspace/localhost/startup/start.sh
./workspaces/t2b_workspace/site_b-1/startup/start.sh
./workspaces/t2b_workspace/site_b-2/startup/start.sh
