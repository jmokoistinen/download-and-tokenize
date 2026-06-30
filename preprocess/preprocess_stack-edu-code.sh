#!/bin/bash
#SBATCH --job-name=preprocess
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --account=amd-tw-verification
#SBATCH --output=logs/preprocess_all-%j.out
#SBATCH --error=logs/preprocess_all-%j.err
#SBATCH --open-mode=append

set -uo pipefail  # removed -e so failures don't abort the whole script

source activate opt-rocm

export PYTHONNOUSERSITE=1
export MEGATRON_DIR=/shared_silo/scratch/mika/experiments/Megatron-Bridge/3rdparty/Megatron-LM
TOKENIZER=/shared_silo/scratch/poro3_tokenizer/mika/models/Qwen3.5-4B-Base #check the tokenizer here and below!

DATASET_DIR=/shared_silo/scratch/ttoivone/datasets/stack-edu
TOKENIZED_DIR=/shared_silo/scratch/ttoivone/tokenized/stack-edu-tokenized
CLEANED_DIR=/shared_silo/scratch/ttoivone/datasets/stack-edu

BRIDGE_ROOT=/shared_silo/scratch/mika/experiments/Megatron-Bridge
source ~/environment.sh #your HF_TOKEN

CONTAINER="/shared_silo/scratch/containers/build-rocm_primus_v25.11_transformers-5.5.4_linear_FA/rocm_primus_v25.11_transformers-5.5.4_linear_FA.sif"
BIND_PATH="${BIND_PATH:-/shared_silo/scratch/}"

failed=()

for INPUTFILE in "$DATASET_DIR"/*.jsonl; do
    # skip already-cleaned files
    [[ "$INPUTFILE" == *_clean.jsonl ]] && continue

    #suffix=$(basename "$INPUTFILE" .jsonl)
    #suffix="${suffix#fin_Latn_}"
    #[[ ! "$suffix" > "0457" ]] && continue

    BASENAME=$(basename "$INPUTFILE" .jsonl)
    CLEAN_FILE="${CLEANED_DIR}/${BASENAME}_clean.jsonl"
    #CLEAN_FILE="${INPUTFILE%.jsonl}_clean.jsonl"
    OUTPUT_PREFIX="$TOKENIZED_DIR/$BASENAME"

    echo "=== Processing: $BASENAME ==="
    mkdir -p "$OUTPUT_PREFIX"

    python -c "
import sys, json
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('/shared_silo/scratch/mika/models/Qwen3.5-4B-Base')
with open('$INPUTFILE', encoding='utf-8', errors='ignore') as fin, open('$CLEAN_FILE', 'w', encoding='utf-8') as fout:
    skipped = 0
    processed = 0
    processed_messages = 0
    processed_text = 0
    processed_content = 0
    for line in fin:
        try:
            json_line = json.loads(line)
            if 'text' not in json_line and 'messages' in json_line:
                text = tokenizer.apply_chat_template(
                    json_line['messages'],
                    tokenize=False,
                    add_generation_prompt=False,
                )
                fout.write(json.dumps({'text': text}) + '\n')
                processed_messages += 1
            elif 'text' not in json_line and 'content' in json_line:
                fout.write(json.dumps({'text': json_line['content']}) + '\n')
                processed_content += 1
            else:
                fout.write(line)
                processed_text += 1
            processed += 1
            if processed % 1000 == 0:
                print(f'Processed {processed} so far [messages key: {processed_messages}, text key: {processed_text}, content key: {processed_content}]')
        except (json.JSONDecodeError, KeyError):
            skipped += 1
    print(f'Skipped {skipped} bad records', file=sys.stderr)
" || { echo "Cleaning failed for $BASENAME"; failed+=("$INPUTFILE"); continue; }

    apptainer exec --rocm \
        -B "${BIND_PATH}:${BIND_PATH}:rw" \
        --env PYTHONPATH="${BRIDGE_ROOT}/python-packages:${BRIDGE_ROOT}/3rdparty/Megatron-LM:${BRIDGE_ROOT}/src" \
        "$CONTAINER" \
        python3 $MEGATRON_DIR/tools/preprocess_data.py \
            --input "$CLEAN_FILE" \
            --output-prefix "$OUTPUT_PREFIX" \
            --tokenizer-type HuggingFaceTokenizer \
            --tokenizer-model "$TOKENIZER" \
            --log-interval 10000 --workers 16 --append-eod \
    || { echo "Preprocessing failed for $BASENAME"; failed+=("$INPUTFILE"); continue; }

    echo "Done: $BASENAME -> $OUTPUT_PREFIX"
done

echo ""
if [ ${#failed[@]} -eq 0 ]; then
    echo "All files processed successfully."
else
    echo "Failed files (${#failed[@]}):"
    for f in "${failed[@]}"; do
        echo "  $f"
    done
fi