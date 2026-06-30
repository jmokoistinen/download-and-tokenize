# Download (HF datasets) 

git clone https://github.com/jmokoistinen/download-and-tokenize.git

cd download-and-tokenize/download 

sbatch download.sh <hf-dataset-path> <split> --output-dir <output_path> --config <subset> (optional) --shuffle (optional) --seed <seed_num> (optional)

*and others

#### 1 Full download [2152112 lines] 

```
sbatch download.sh allenai/Dolci-Instruct-SFT train --output-dir ./datasets/
``` 

#### Full download with config/subset [561598 lines]

```
sbatch download.sh wikimedia/wikipedia train --config 20231101.fi --output-dir ./datasets/
```

#### 2 Token cap [1480150 tokens] (the main idea was to download max 500B = 5e

```
sbatch download.sh allenai/c4 train --config en --max-tokens 2e6 --output-dir ./datasets/
```

#### Token cap + shuffle (random seed, printed in log) 

```
sbatch download.sh allenai/c4 train --config en --max-tokens 1e8 --shuffle --output-dir ./datasets/
```

#### Token cap + shuffle with fixed seed

```
sbatch download.sh allenai/c4 train --config en --max-tokens 5e9 --shuffle --seed 42 --output-dir ./datasets
```

#### 3 Nemotron math with config 

```
sbatch download.sh nvidia/Nemotron-CC-Math-v1 train --config 4plus_MIND --output-dir ./datasets/
```

#### 4 10% slice 

```
sbatch download.sh allenai/c4 train --config en --sample-rate 0.1 --output-dir /datasets/c4
```




# Tokenize
cd download_and_tokenize/preprocess/ 
eos_token = <|endoftext|>

#### Standard 

Define the folder and other parameters in the file 'download_and_tokenize/preprocess/tokenize_all_data.sh', processes all files in the whole folder. Processes also chat data with template <|im_start|> <|im_end|>

*BUG some chat datasets have tags inside in jsonl format, so it can cause unequal number of tags ( <think> and </think> ...)


#### Finephrase dataset has the rollouts and those are taken instead of text, 

```
sbatch preprocess.sh /shared_silo/scratch/datasets/english/finephrase/FinePhrase-faq-max-125Btok.jsonl /shared_silo/scratch/datasets/tokenized/qwen3.5/megatron/finephrase/FinePhrase-faq-train

sbatch preprocess.sh /shared_silo/scratch/datasets/english/finephrase/FinePhrase-math-max-125Btok.jsonl /shared_silo/scratch/datasets/tokenized/qwen3.5/megatron/finephrase/FinePhrase-math-train

sbatch preprocess.sh /shared_silo/scratch/datasets/english/finephrase/FinePhrase-table-max-125Btok.jsonl /shared_silo/scratch/datasets/tokenized/qwen3.5/megatron/finephrase/FinePhrase-table-train

sbatch preprocess.sh /shared_silo/scratch/datasets/english/finephrase/FinePhrase-tutorial-max-125Btok.jsonl /shared_silo/scratch/datasets/tokenized/qwen3.5/megatron/finephrase/FinePhrase-tutorial-train
```

*BUG there is empty lines here from empty rollout lines, take out!

*QUESTION is there spercific processing needed here with the rollouts? 


