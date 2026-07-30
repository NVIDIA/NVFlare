#!/bin/bash
#SBATCH --account=coreai_edgeai_flresearch
#SBATCH --partition=polar,polar3,polar4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=01:30:00
#SBATCH --job-name=collab-qwen3-8b-single
#SBATCH --output=/lustre/fsw/portfolios/coreai/users/ziyuex/projects/collab_project/logs/%x_%j.out
#SBATCH --error=/lustre/fsw/portfolios/coreai/users/ziyuex/projects/collab_project/logs/%x_%j.err

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec "${SCRIPT_DIR}/run_qwen3_8b.sh" single
