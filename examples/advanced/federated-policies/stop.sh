#!/usr/bin/env bash
ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
DIR=${ROOT}/workspace/fed_policy/prod_00
echo "Stopping clients ..."
echo "y" | ${DIR}/site_a/startup/stop_fl.sh
echo "y" | ${DIR}/site_b/startup/stop_fl.sh

echo "Stopping server ..."
echo "y" | ${DIR}/server1/startup/stop_fl.sh

sleep 10
echo "Cleaning up ..."
pkill -9 -f federated-policies
echo "NVFlare is shutdown"
