#!/bin/bash
#SBATCH --account=coreai_edgeai_flresearch
#SBATCH --partition=polar,polar3,polar4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=02:00:00
#SBATCH --job-name=collab-qwen3-8b-capacity
#SBATCH --output=/lustre/fsw/portfolios/coreai/users/ziyuex/projects/collab_project/logs/%x_%j.out
#SBATCH --error=/lustre/fsw/portfolios/coreai/users/ziyuex/projects/collab_project/logs/%x_%j.err

set -euo pipefail

: "${NVFLARE_SOURCE_ROOT:?Export NVFLARE_SOURCE_ROOT when submitting}"
exec "${NVFLARE_SOURCE_ROOT}/examples/advanced/collab/benchmarks/slurm/run_qwen3_8b.sh" capacity
