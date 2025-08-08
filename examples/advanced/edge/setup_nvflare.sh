#!/bin/bash
PROJECT_NAME=edge_example
PROV_SCRIPT="../../../nvflare/edge/tools/tree_prov.py"
WORKSPACE_DIR="/tmp/nvflare/workspaces"

# Check if project directory exists and remove it
if [ -d "$WORKSPACE_DIR/$PROJECT_NAME" ]; then
    rm -rf "$WORKSPACE_DIR/$PROJECT_NAME"
fi

python "$PROV_SCRIPT" --root_dir $WORKSPACE_DIR -p $PROJECT_NAME -d 1 -w 2
