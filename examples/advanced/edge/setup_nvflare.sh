#!/bin/bash
PROJECT_NAME=edge_example
WORKSPACE_DIR="/tmp/nvflare/workspaces"

# Check if project directory exists and remove it
if [ -d "$WORKSPACE_DIR" ]; then
    rm -rf "$WORKSPACE_DIR"
fi

# Create the workspace directory and copy the project.yml file
mkdir -p $WORKSPACE_DIR
cp project.yml $WORKSPACE_DIR

# Provision the project with the project.yml
cd $WORKSPACE_DIR
nvflare provision -p project.yml