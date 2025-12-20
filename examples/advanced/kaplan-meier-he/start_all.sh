#!/bin/bash
# Convenience script to start all parties for local production environment testing
# This starts the server and all 5 clients in the background

set -e

WORKSPACE_ROOT="/tmp/nvflare/prod_workspaces/km_he_project/prod_00"

echo "======================================"
echo "Starting NVFlare Production Environment"
echo "======================================"
echo ""

# Check if workspace exists
if [ ! -d "$WORKSPACE_ROOT" ]; then
    echo "Error: Workspace not found at $WORKSPACE_ROOT"
    echo "Please run provisioning first:"
    echo "  nvflare provision -p project.yml -w /tmp/nvflare/prod_workspaces"
    exit 1
fi

# Function to start a component
start_component() {
    local name=$1
    local path=$2
    local log_file="/tmp/nvflare/logs/${name}.log"
    
    mkdir -p /tmp/nvflare/logs
    
    echo "Starting ${name}..."
    cd "${path}"
    ./startup/start.sh > "${log_file}" 2>&1 &
    local pid=$!
    echo "  PID: ${pid}"
    echo "  Log: ${log_file}"
    echo "${pid}" > "/tmp/nvflare/logs/${name}.pid"
}

# Start server
echo ""
echo "1. Starting Server..."
start_component "localhost" "${WORKSPACE_ROOT}/localhost"
echo "   Waiting for server to be ready..."
sleep 10

# Start clients
echo ""
echo "2. Starting Clients..."
for i in {1..5}; do
    start_component "site-${i}" "${WORKSPACE_ROOT}/site-${i}"
    sleep 2
done

echo ""
echo "======================================"
echo "All parties started successfully!"
echo "======================================"
echo ""
echo "Server and clients are running in the background."
echo "Logs are available in /tmp/nvflare/logs/"
echo ""
echo "To check status:"
echo "  tail -f /tmp/nvflare/logs/localhost.log"
echo "  tail -f /tmp/nvflare/logs/site-1.log"
echo ""
echo "To start admin console:"
echo "  cd ${WORKSPACE_ROOT}/admin@nvidia.com"
echo "  ./startup/fl_admin.sh"
echo ""
echo "To stop all parties, use Admin Console"
echo ""

