#!/usr/bin/env bash
set -euo pipefail
export NVFL_WORKSPACE=@@NVFLARE_WORKSPACE_PATH@@
exec "$NVFL_WORKSPACE/startup/sub_start.sh" --once
