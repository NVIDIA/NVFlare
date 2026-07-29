#!/usr/bin/env bash
@@NVFLARE_SBATCH_DIRECTIVES@@set -euo pipefail
@@NVFLARE_ENVIRONMENT_SETUP@@WORKSPACE_ROOT=@@NVFLARE_WORKSPACE_ROOT@@
exec "$WORKSPACE_ROOT/startup/start_slurm.sh"
