#!/bin/bash
PROJECT_NAME=edge_example
WORKSPACE_DIR="/tmp/nvflare/workspaces"

# Change to the edge_example/prod_00 directory
cd $WORKSPACE_DIR/$PROJECT_NAME/prod_00 || exit 1

# Start the NVFlare system
./start_all.sh
