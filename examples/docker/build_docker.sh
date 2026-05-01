#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

docker build -t nvflare-site:latest -f "$SCRIPT_DIR/Dockerfile" "$REPO_ROOT"
docker build -t nvflare-job:latest -f "$SCRIPT_DIR/Dockerfile.nvflare-job" "$REPO_ROOT"
