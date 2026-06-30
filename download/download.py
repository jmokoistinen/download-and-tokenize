#!/usr/bin/env python3
"""
Download HuggingFace datasets to JSONL with optional filtering.

Three download modes (auto-selected based on flags):


1. Full download (no flags) — parallel, fastest: [useful for smaller datasets]
        python3 download_basic.py allenai/Dolci-Instruct-SFT train --output-dir /datasets/chat/Dolci

2. Token cap (--max-tokens) and/or shuffle (--shuffle) — streaming, slower: 
        python3 download_basic.py allenai/c4 train --config bg --max-tokens 5e9 --output-dir /datasets/c4
        # stops at ~5B tokens → c4-en-max-5Btok.jsonl

        python3 download_basic.py allenai/c4 train --config bg --max-tokens 5e9 --shuffle --seed 42 --output-dir /datasets/c4
        # shuffles then caps → same filename but randomised rows

3. Percentage slice (--sample-rate) — parallel, fast:
        python3 download_basic.py allenai/c4 train --config bg --sample-rate 0.1 --output-dir /datasets/c4
        # takes first 10% → c4-en-10pct.jsonl


4. Dataset subsets (--config): [already used in the 2 and 3.]
        python3 download_basic.py wikimedia/wikipedia train --config 20231101.fi --output-dir /datasets/wiki
        # → wikipedia-20231101.fi.jsonl

* if seed is not passed it will have random seed 

Output filename encodes what was done:
dataset-split.jsonl               full
dataset-split-10pct.jsonl         10% slice
dataset-split-max-5Btok.jsonl     5B token cap
dataset-config-max-500Mtok.jsonl  config + 500M token cap
"""

from os import path
from pathlib import Path
from argparse import ArgumentParser
import json
from datasets import load_dataset
import random as _random

ap = ArgumentParser()
ap.add_argument('name')
ap.add_argument('split')
ap.add_argument('--sample-rate', type=float, default=None, help='e.g. 0.1 = 10%')
ap.add_argument('--max-tokens', type=float, default=None, help='e.g. 500000000 = 0.5B')
ap.add_argument('--config', default=None, help='dataset config/subset, e.g. 20231101.fi')
ap.add_argument('--output-dir', default='.')
ap.add_argument('--shuffle', action='store_true')
ap.add_argument('--seed', type=int, default=None, help='random seed (default: random)')
args = ap.parse_args()

out_dir = Path(args.output_dir).resolve()
out_dir.mkdir(parents=True, exist_ok=True)

# Different savename suffix for different runs
if args.max_tokens:
    tok = args.max_tokens
    suffix = f'-max-{int(tok/1e9)}Btok' if tok >= 1e9 else f'-max-{int(tok/1e6)}Mtok'
elif args.sample_rate:
    suffix = f'-{int(args.sample_rate * 100)}pct'
else:
    suffix = ''
if args.shuffle:
    suffix += '-shuffled'    

out_name = str(out_dir / f'{path.basename(args.name)}-{args.config or args.split}{suffix}.jsonl')
print(f'Output: {out_name}', flush=True)

# Simple percentage slice — no streaming needed, fast
if args.sample_rate and not args.max_tokens:
    pct = int(args.sample_rate * 100)
    ds = load_dataset(args.name, args.config, split=f'{args.split}[:{pct}%]', num_proc=16)
    ds.to_json(out_name)

# Full download — no filtering, fast parallel path
elif not args.max_tokens and not args.shuffle:
    ds = load_dataset(args.name, args.config, split=args.split, num_proc=32)
    print(f'Rows: {len(ds):,}', flush=True)
    ds.to_json(out_name)
    print(f'Done. {len(ds):,} rows → {out_name}')

# Token cap or shuffle — requires streaming
else:
    ds = load_dataset(args.name, args.config, split=args.split, streaming=True)
    if args.shuffle:
        seed = args.seed if args.seed is not None else _random.randint(0, 2**32)
        print(f'Shuffle seed: {seed}', flush=True)
        ds = ds.shuffle(seed=seed, buffer_size=10_000)

    max_tok = args.max_tokens or float('inf')
    total_tokens = 0
    total_rows = 0

    with open(out_name, 'w', encoding='utf-8') as f:
        for row in ds:
            text = row.get('text', '')
            tok = len(text.split()) * 1.3
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
            total_tokens += tok
            total_rows += 1
            if total_rows % 10_000 == 0:
                print(f'  {total_rows:,} rows | {total_tokens/1e9:.3f}B tokens', flush=True)
            if total_tokens >= max_tok:
                break

    print(f'Done. {total_rows:,} rows | {total_tokens/1e9:.3f}B tokens → {out_name}')