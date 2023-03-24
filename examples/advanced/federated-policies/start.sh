#!/usr/bin/env bash
ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
DIR=${ROOT}/workspace/fed_policy/prod_00
echo "Starting server ..."
${DIR}/server1/startup/start.sh
sleep 10

echo "Starting clients ..."
${DIR}/site_a/startup/start.sh
${DIR}/site_b/startup/start.sh
