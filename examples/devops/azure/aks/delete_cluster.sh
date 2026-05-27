#!/usr/bin/env bash

set -euo pipefail

RESOURCE_GROUP="${RESOURCE_GROUP:-myResourceGroup}"

az group delete --name "${RESOURCE_GROUP}" --yes
