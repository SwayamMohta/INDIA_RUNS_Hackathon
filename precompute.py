"""
precompute.py — Offline precomputation (run once, no time limit).

Performs all expensive operations that don't need to happen within
the 5-minute ranking window:
  1. Parse all 100k candidates from JSONL
  2. Build BM25 index on career descriptions
  3. Compute all features for every candidate
  4. Save feature matrix as parquet + BM25 index as pickle

Estimated runtime: 10-20 minutes for 100k candidates.
Estimated output size: ~200-300MB (feature parquet + BM25 pickle)

Usage:
    python precompute.py [--candidates candidates.jsonl] [--output data/]
"""

import argparse
import json
import os
import pickle
import sys
import time
from typing import Dict, List, Any

import pandas as pd
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ranker.loader import stream_candidates, load_sample_candidates
from ranker.bm25_retrieval import build_bm25_index, save_bm25_artifacts
from ranker.feature_frame import build_feature_frame


def parse_args():
    parser = argparse.ArgumentParser(description="Redrob Ranker — Precomputation")
    parser.add_argument(
        "--candidates", default="candidates.jsonl",
        help="Path to candidates JSONL file (default: candidates.jsonl)"
    )
    parser.add_argument(
        "--sample", action="store_true",
        help="Use sample_candidates.json instead (for testing)"
    )
    parser.add_argument(
        "--output", default="data",
        help="Output directory for precomputed artifacts (default: data/)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    t_start = time.time()

    # ── Step 1: Load candidates ──────────────────────────────────
    if args.sample:
        sample_path = args.candidates if args.candidates != "candidates.jsonl" else "sample_candidates.json"
        print(f"[precompute] Loading sample candidates ({sample_path})...")
        candidates = load_sample_candidates(sample_path)
        print(f"[precompute] Loaded {len(candidates)} sample candidates.")
    else:
        print(f"[precompute] Streaming candidates from {args.candidates}...")
        candidates = list(tqdm(
            stream_candidates(args.candidates),
            desc="Loading candidates",
            unit="cand"
        ))
        print(f"[precompute] Loaded {len(candidates)} candidates in {time.time()-t_start:.1f}s")

    # ── Step 2: Build BM25 index ─────────────────────────────────
    print("\n[precompute] Building BM25 index...")
    t_bm25 = time.time()
    bm25_model, candidate_ids = build_bm25_index(candidates, verbose=True)
    save_bm25_artifacts(bm25_model, candidate_ids, args.output)
    print(f"[precompute] BM25 index built in {time.time()-t_bm25:.1f}s")

    # ── Step 3: Compute features ─────────────────────────────────
    # Uses the SAME builder as rank.py's standalone path (ranker/feature_frame.py),
    # so precomputed and on-the-fly features are constructed identically.
    print("\n[precompute] Computing features for all candidates...")
    t_feats = time.time()
    df = build_feature_frame(candidates, show_progress=True)
    print(f"[precompute] Features computed in {time.time()-t_feats:.1f}s")

    # ── Step 4: Save feature matrix ──────────────────────────────
    print("\n[precompute] Saving feature matrix...")

    # Save as parquet for fast loading
    parquet_path = os.path.join(args.output, "features.parquet")
    df.to_parquet(parquet_path)
    print(f"[precompute] Saved features to {parquet_path} ({os.path.getsize(parquet_path)/1e6:.1f} MB)")

    # Also save candidate_ids list for BM25 alignment
    ids_path = os.path.join(args.output, "candidate_ids.json")
    with open(ids_path, "w") as f:
        json.dump(candidate_ids, f)

    t_total = time.time() - t_start
    print(f"\n[precompute] DONE in {t_total:.1f}s ({t_total/60:.1f} min)")
    print(f"[precompute] Artifacts saved to: {os.path.abspath(args.output)}/")
    print(f"  - {args.output}/bm25_index.pkl")
    print(f"  - {args.output}/features.parquet")
    print(f"  - {args.output}/candidate_ids.json")


if __name__ == "__main__":
    main()
