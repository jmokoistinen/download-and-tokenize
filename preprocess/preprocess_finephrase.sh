#!/bin/bash
#SBATCH --job-name=preprocess
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --account=amd-tw-verification
#SBATCH --output logs/%j.out
#SBATCH --error logs/%j.err
#SBATCH --open-mode=append
#SBATCH --output=logs/preprocess-%j.out
#SBATCH --error=logs/preprocess-%j.err

set -euo pipefail

source ~/environment.sh  #your HF_TOKEN to load tokenizer from hf etc

export PYTHONNOUSERSITE=1 #Python won’t add the user site-packages directory to sys.path.
export MEGATRON_DIR=/shared_silo/scratch/mika/experiments/Megatron-Bridge/3rdparty/Megatron-LM
TOKENIZER=/shared_silo/scratch/mika/models/Qwen3.5-4B-Base #check the tokenizer here and below!

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 INPUTFILE OUTPUT_PREFIX" >&2
    exit 1
fi

INPUTFILE="$1"
OUTPUT_PREFIX="$2"

echo "Command: $0 $*"
echo "Input:   $INPUTFILE"
echo "Output:  $OUTPUT_PREFIX"
echo "Started: $(date)"


mkdir -p $OUTPUT_PREFIX

#set your own path here
BRIDGE_ROOT=/shared_silo/scratch/mika/experiments/Megatron-Bridge  #

CONTAINER="/shared_silo/scratch/containers/build-rocm_primus_v25.11_transformers-5.5.4_linear_FA/rocm_primus_v25.11_transformers-5.5.4_linear_FA.sif"
BIND_PATH="${BIND_PATH:-/shared_silo/scratch/}" #note

#apptainer shell --rocm \
#    -B "${BIND_PATH}:${BIND_PATH}:rw" \
#    --env PYTHONPATH="${BRIDGE_ROOT}/python-packages:${BRIDGE_ROOT}/3rdparty/Megatron-LM:${BRIDGE_ROOT}/src" \
#    "$CONTAINER" 

apptainer exec --rocm \
    -B "${BIND_PATH}:${BIND_PATH}:rw" \
    --env PYTHONPATH="${BRIDGE_ROOT}/python-packages:${BRIDGE_ROOT}/3rdparty/Megatron-LM:${BRIDGE_ROOT}/src" \
    "$CONTAINER" \
    python3 preprocess_data_finephrase.py \
        --input $INPUTFILE \
        --output-prefix $OUTPUT_PREFIX \
        --tokenizer-type HuggingFaceTokenizer \
        --tokenizer-model /shared_silo/scratch/mika/models/Qwen3.5-4B-Base \
        --rollout-key rollout_results \
        --json-keys text \
        --log-interval 10000 --workers 32 --append-eod 
#16

echo "Done: $(date)"
echo "Input:  $INPUTFILE"
echo "Done. tokenized data stored at Output: $OUTPUT_PREFIX"

