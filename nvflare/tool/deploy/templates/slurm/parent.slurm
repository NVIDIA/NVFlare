#!/usr/bin/env bash
@@NVFLARE_SBATCH_DIRECTIVES@@set -euo pipefail
@@NVFLARE_ENVIRONMENT_SETUP@@PREPARED_ROOT=@@NVFLARE_PREPARED_ROOT@@
exec "$PREPARED_ROOT/startup/start_slurm.sh"
