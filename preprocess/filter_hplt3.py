#!/usr/bin/env python3
"""
Filter locally downloaded HPLT3.0 JSONL files.

Usage:
python filter_hplt3.py \
            --input-dir \
            ./fin_hplt3_0.001 \
            --output-dir \
            ./fin_hplt3_0.001_filtered_8 \
            --lang-purity 0.0 \
            --min-lang-prob 0.9 \
            --min-doc-score 7.5 \
            --max-zero-fraction 0.5 \
            --max-cluster-size 5 \
            --max-mt-score 0.3 \
            --min-words 50 \
            --filter-pii           
"""
##

import json
import argparse
import statistics
from pathlib import Path
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Blocklists / signal lists
# ---------------------------------------------------------------------------

ADULT_DOMAINS = {
    "pornhub.com", "xvideos.com", "xnxx.com", "xhamster.com",
    "redtube.com", "youporn.com", "tube8.com", "spankbang.com",
    "chaturbate.com", "onlyfans.com", "brazzers.com", "bangbros.com",
}

CREDENTIAL_SIGNALS = [
    "password=", "passwd=", "pass=",
    "api_key=", "api-key=", "apikey=",
    "secret_key=", "secret=",
    "access_token=", "auth_token=", "bearer ",
    "private_key", "-----begin rsa", "-----begin ec",
    "aws_access_key_id", "aws_secret_access_key",
    "client_secret=", "database_url=", "db_password=",
    "smtp_password=", "ftp_password=",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_adult_url(url: str) -> bool:
    try:
        domain = urlparse(url).netloc.lower().lstrip("www.")
        return any(domain == d or domain.endswith("." + d) for d in ADULT_DOMAINS)
    except Exception:
        return False


def has_credentials(text: str) -> bool:
    t = text.lower()
    return any(sig in t for sig in CREDENTIAL_SIGNALS)


def mean_doc_score(doc_scores: list) -> float:
    return statistics.mean(doc_scores) if doc_scores else 0.0


def zero_fraction(doc_scores: list) -> float:
    if not doc_scores:
        return 1.0
    return sum(1 for s in doc_scores if s == 0) / len(doc_scores)


def seg_purity(seg_langs: list, target: str) -> float:
    if not seg_langs:
        return 0.0
    return sum(1 for s in seg_langs if s == target) / len(seg_langs)


def estimate_tokens(text: str) -> float:
    # Finnish on Qwen BPE: ~2 tokens/word (agglutinative morphology)
    return len(text.split()) * 2.0


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------

def passes_filters(row: dict, args) -> tuple[bool, str]:
    """Returns (keep, rejection_reason)."""

    # 1. Upstream HPLT pipeline decision
    if row.get("filter") != "keep":
        return False, "upstream_filter"

    # 2. Document-level language confidence (OpenLID)
    langs = row.get("lang", [])
    probs = row.get("prob", [])
    lang_prob = probs[0] if (langs and probs and langs[0] == args.lang) else 0.0
    if lang_prob < args.min_lang_prob:
        return False, "lang_prob"

    # 3. Segment-level language purity
    #    BUG FIX: field is "seg_langs" not "seg_lang"
    segs = row.get("seg_langs", [])
    if segs and seg_purity(segs, args.lang) < args.lang_purity:
        return False, "seg_purity"

    # 4. WDS quality — mean of per-segment doc_scores (scale 0–10)
    doc_scores = row.get("doc_scores", [])
    if mean_doc_score(doc_scores) < args.min_doc_score:
        return False, "doc_score"

    # 5. Zero-score fraction — high fraction signals structurally broken doc
    #    (short structured lists score 0 legitimately; only reject if majority are 0)
    if zero_fraction(doc_scores) > args.max_zero_fraction:
        return False, "zero_fraction"

    # 6. Near-duplicate cluster size
    if row.get("cluster_size", 1) > args.max_cluster_size:
        return False, "cluster_size"

    # 7. Machine-translated content
    mt = row.get("web-register", {}).get("MT", 0.0)
    if mt > args.max_mt_score:
        return False, "machine_translated"

    # 8. Adult content — URL domain blocklist
    if is_adult_url(row.get("u", "")):
        return False, "adult_url"

    # 9. PII annotations (emails, phone numbers etc. found by HPLT pipeline)
    if args.filter_pii and row.get("pii"):
        return False, "pii"

    text = row.get("text", "")

    # 10. Credentials — heuristic scan of text content
    if has_credentials(text):
        return False, "credentials"

    # 11. Minimum document length
    if len(text.split()) < args.min_words:
        return False, "too_short"

    return True, ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir",         required=True,
                    help="Directory containing downloaded HPLT3 JSONL files")
    p.add_argument("--output-dir",        required=True,
                    help="Output directory for filtered JSONL files")
    p.add_argument("--lang",              default="fin_Latn",
                    help="Target language code (default: fin_Latn)")
    p.add_argument("--min-lang-prob",     type=float, default=0.9,
                    help="Min OpenLID document-level language probability (default: 0.9)")
    p.add_argument("--lang-purity",       type=float, default=0.75,
                    help="Min fraction of seg_langs matching target lang (default: 0.75)")
    p.add_argument("--min-doc-score",     type=float, default=7.0,
                    help="Min mean WDS doc_score 0–10 (default: 7.0)")
    p.add_argument("--max-zero-fraction", type=float, default=0.5,
                    help="Max fraction of doc_scores that are exactly 0 (default: 0.5)")
    p.add_argument("--max-cluster-size",  type=int,   default=5,
                    help="Max near-duplicate cluster size (default: 5)")
    p.add_argument("--max-mt-score",      type=float, default=0.3,
                    help="Max web-register MT probability (default: 0.3)")
    p.add_argument("--filter-pii",        action="store_true",
                    help="Reject documents with any PII annotations")
    p.add_argument("--min-words",         type=int,   default=50,
                    help="Minimum document word count (default: 50)")
    p.add_argument("--rows-per-file",     type=int,   default=100_000,
                    help="Output rows per shard file (default: 100000)")
    args = p.parse_args()

    in_dir  = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(in_dir.glob("*.jsonl"))
    if not input_files:
        print(f"No .jsonl files found in {in_dir}")
        return

    print(f"Input files:       {len(input_files)}")
    print(f"Lang:              {args.lang}")
    print(f"min_lang_prob:     {args.min_lang_prob}")
    print(f"lang_purity:       {args.lang_purity}")
    print(f"min_doc_score:     {args.min_doc_score}")
    print(f"max_zero_fraction: {args.max_zero_fraction}")
    print(f"max_cluster_size:  {args.max_cluster_size}")
    print(f"max_mt_score:      {args.max_mt_score}")
    print(f"filter_pii:        {args.filter_pii}")
    print(f"min_words:         {args.min_words}")
    print()

    counters = {
        "total": 0, "kept": 0,
        "upstream_filter": 0, "lang_prob": 0, "seg_purity": 0,
        "doc_score": 0, "zero_fraction": 0, "cluster_size": 0,
        "machine_translated": 0, "adult_url": 0, "pii": 0,
        "credentials": 0, "too_short": 0,
    }

    shard_idx     = 0
    rows_in_shard = 0
    total_tokens  = 0.0
    out_f         = None

    for file_idx, in_path in enumerate(input_files, 1):
        print(f"[{file_idx}/{len(input_files)}] {in_path.name}", flush=True)

        with open(in_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue

                counters["total"] += 1
                keep, reason = passes_filters(row, args)

                if not keep:
                    counters[reason] += 1
                    continue

                if out_f is None:
                    out_path = out_dir / f"{args.lang}_filtered_{shard_idx:04d}.jsonl"
                    out_f = open(out_path, "w", encoding="utf-8")

                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                counters["kept"]  += 1
                rows_in_shard     += 1
                total_tokens      += estimate_tokens(row.get("text", ""))

                if rows_in_shard >= args.rows_per_file:
                    out_f.close()
                    out_f         = None
                    rows_in_shard = 0
                    shard_idx     += 1
        pct = 100 * counters["kept"] / max(counters["total"], 1)
        print(f"  → {counters['total']:,} processed | {counters['kept']:,} kept ({pct:.1f}%) |{total_tokens/1e9:.2f}B tok", flush=True)

    if out_f:
        out_f.close()

    # Summary
    total = counters["total"]
    kept  = counters["kept"]
    print(f"\n{'='*55}")
    print(f"Total processed :  {total:>12,}")
    print(f"Kept            :  {kept:>12,}  ({100*kept/max(total,1):.1f}%)")
    print(f"Tokens (est.)   :  {total_tokens/1e9:>11.2f}B")
    print(f"Output shards   :  {shard_idx + (1 if rows_in_shard else 0)}")
    print(f"\nRejection breakdown:")
    skip_keys = {"total", "kept"}
    for reason, cnt in sorted(counters.items()):
        if reason in skip_keys or cnt == 0:
            continue
        bar = "█" * int(30 * cnt / max(total, 1))
        print(f"  {reason:<22} {cnt:>10,}  ({100*cnt/max(total,1):5.1f}%)  {bar}")


if __name__ == "__main__":
    main()
