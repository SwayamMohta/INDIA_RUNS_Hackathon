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
from ranker.features import compute_all_features
from ranker.honeypot import get_trap_multipliers


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
    print("\n[precompute] Computing features for all candidates...")
    t_feats = time.time()

    all_feature_rows = []
    failed = 0

    for cand in tqdm(candidates, desc="Computing features", unit="cand"):
        cid = cand.get("candidate_id", "")
        try:
            feats = compute_all_features(cand)
            hp_mult, st_mult, trap_reason = get_trap_multipliers(cand)
            feats["candidate_id"] = cid
            feats["honeypot_multiplier"] = hp_mult
            feats["stuffer_multiplier"] = st_mult
            feats["trap_reason"] = trap_reason

            # Store raw fields needed for reasoning generation
            profile = cand.get("profile", {})
            feats["_yoe"] = profile.get("years_of_experience", 0)
            feats["_current_title"] = profile.get("current_title", "")
            feats["_current_company"] = profile.get("current_company", "")
            feats["_location"] = profile.get("location", "")
            feats["_rrr"] = cand.get("redrob_signals", {}).get("recruiter_response_rate", 0)
            feats["_notice"] = cand.get("redrob_signals", {}).get("notice_period_days", 60)
            feats["_otw"] = int(cand.get("redrob_signals", {}).get("open_to_work_flag", False))

            # Cache top skills for reasoning generation
            skills = cand.get("skills", [])
            top_skills = [
                s.get("name", "") for s in skills
                if s.get("proficiency") in ("expert", "advanced") and s.get("endorsements", 0) > 5
            ][:4]
            feats["_top_skills"] = ", ".join(top_skills) if top_skills else ""

            # Cache top product company for reasoning
            career = cand.get("career_history", [])
            from ranker.features import _classify_company
            product_cos = [
                r.get("company", "") for r in career
                if _classify_company(r.get("company", "")) in ("tier1", "tier2", "tier3")
                and r.get("company", "") not in ("", "Unknown")
            ]
            feats["_top_company"] = product_cos[0] if product_cos else profile.get("current_company", "")

            all_feature_rows.append(feats)
        except Exception as e:
            failed += 1
            if failed <= 10:
                print(f"[precompute] WARNING: failed on {cid}: {e}")
            all_feature_rows.append({"candidate_id": cid, "error": str(e)})

    print(f"[precompute] Features computed in {time.time()-t_feats:.1f}s ({failed} failures)")

    # ── Step 4: Save feature matrix ──────────────────────────────
    print("\n[precompute] Saving feature matrix...")
    df = pd.DataFrame(all_feature_rows)
    df = df.set_index("candidate_id")

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
