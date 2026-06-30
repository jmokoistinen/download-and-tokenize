#!/bin/bash
#SBATCH --job-name=download
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --account=amd-tw-verification
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail

source ~/environment.sh

SCRIPT_DIR=/shared_silo/scratch/mika/experiments/dataset/download-and-tokenize/download #set to your path
mkdir -p "$SCRIPT_DIR/logs" 
cd "$SCRIPT_DIR"

echo "START $(date)"
echo "Node: ${SLURMD_NODENAME:-local}"
echo "python3 download.py $@"

python3 download.py "$@"

echo "END $(date)"


