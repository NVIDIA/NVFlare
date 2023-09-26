#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
SERVER_MODEL_DIR="$DIR/models/server"
sed -e "s~\$SERVER_MODEL_DIR~$SERVER_MODEL_DIR~g" cifar10_fedavg/config/environment.json.tmp > cifar10_fedavg/config/environment.json

CLIENT_MODEL_DIR="$DIR/models/client"
sed -e "s~\$CLIENT_MODEL_DIR~$CLIENT_MODEL_DIR~g" cifar10_fedavg/config/config_fed_client.json.tmp > cifar10_fedavg/config/config_fed_client.json
