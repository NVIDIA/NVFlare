#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

eksctl create cluster -f "${SCRIPT_DIR}/cluster.yaml"
