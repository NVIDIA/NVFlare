#!/bin/bash
PROJECT_NAME=edge_example
WORKSPACE_DIR="/tmp/nvflare/workspaces"

# Check if project directory exists and remove it
if [ -d "$WORKSPACE_DIR" ]; then
    rm -rf "$WORKSPACE_DIR"
fi

mkdir -p $WORKSPACE_DIR
cd $WORKSPACE_DIR
# Provision the project yaml file
nvflare provision -e
# Update depth, width, and name in project.yml after provisioning
if [ -f "$WORKSPACE_DIR/project.yml" ]; then
    echo "Updating depth and width in project.yml..."
    sed -i 's/depth: 2/depth: 1/' "$WORKSPACE_DIR/project.yml"
    sed -i 's/width: 3/width: 2/' "$WORKSPACE_DIR/project.yml"
    echo "Updated project.yml with depth: 1, width: 2"
    sed -i 's/name: example_project/name: edge_example/' "$WORKSPACE_DIR/project.yml"
    echo "Updated project.yml with name: edge_example"
else
    echo "Warning: project.yml not found at $WORKSPACE_DIR/project.yml"
fi
# Provision the actual project with the updated project.yml
nvflare provision -p project.yml