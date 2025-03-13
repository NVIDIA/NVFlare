#!/bin/bash
PROJECT_NAME=edge_example
PROV_SCRIPT="../../../nvflare/edge/tree_prov.py"
WORKSPACE_DIR="/tmp/nvflare/workspaces"

# Check if project directory exists and remove it
if [ -d "$WORKSPACE_DIR/$PROJECT_NAME" ]; then
    rm -rf "$WORKSPACE_DIR/$PROJECT_NAME"
fi

python "$PROV_SCRIPT" --root_dir $WORKSPACE_DIR -p $PROJECT_NAME -d 1 -w 2

# Define leaf clients array
leaf_clients=(C11 C12 C21 C22)

# Loop through leaf clients and copy resources
for client in "${leaf_clients[@]}"; do
    cp edge__p_resources.json $WORKSPACE_DIR/$PROJECT_NAME/prod_00/$client/local/
done

