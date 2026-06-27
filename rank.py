"""
rank.py — Main entry point for the 5-minute ranking pipeline.

This script:
  1. Loads precomputed BM25 index and feature matrix from disk
  2. Runs BM25 scoring over all 100,000 candidates
  3. Applies the full LTR scoring formula to all candidates
  4. Generates template-based reasoning for top-100
  5. Outputs submission.csv

Must complete in < 5 minutes on CPU with 16GB RAM.
No network calls. No GPU.

Usage:
    python rank.py [--candidates candidates.jsonl] [--data data/] [--output submission.csv]

Note:
    If precomputed artifacts don't exist, falls back to computing
    features on-the-fly (slower, may approach time limit for 100k candidates).
"""

import argparse
import csv
import json
import os
import sys
import time
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm  # type: ignore

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ranker.config import BM25_TOP_K, FINAL_TOP_N, JD_QUERY, WEIGHTS
from ranker.loader import stream_candidates, load_sample_candidates
from ranker.bm25_retrieval import query_bm25, load_bm25_artifacts
from ranker.scorer import (
    score_candidate,
    compute_career_trajectory_score,
    compute_technical_fit_score,
    compute_behavioral_score,
    compute_availability_score,
    compute_education_score,
)
from ranker.reasoning import generate_reasoning, truncate_reasoning
from ranker.config import CONSULTING_ONLY_MULTIPLIER, DISQUALIFYING_TITLE_MULTIPLIER, DISQUALIFYING_TITLES, SUSPICIOUS_HONEYPOT_THRESHOLD


def parse_args():
    parser = argparse.ArgumentParser(description="Redrob Ranker — Main Ranking Pipeline")
    parser.add_argument(
        "--candidates", default="candidates.jsonl",
        help="Path to candidates JSONL or JSON file (default: candidates.jsonl)"
    )
    parser.add_argument(
        "--sample", action="store_true",
        help="Use sample_candidates.json instead (for testing)"
    )
    parser.add_argument(
        "--data", default="data",
        help="Directory containing precomputed artifacts (default: data/)"
    )
    parser.add_argument(
        "--output", "--out", default="submission.csv",
        help="Output CSV path (default: submission.csv)"
    )
    parser.add_argument(
        "--top-k-bm25", type=int, default=BM25_TOP_K,
        help=f"Number of candidates from BM25 retrieval (default: {BM25_TOP_K})"
    )
    return parser.parse_args()


def load_precomputed(data_dir: str):
    """Load precomputed BM25 index and feature matrix."""
    bm25_path = os.path.join(data_dir, "bm25_index.pkl")
    feats_path = os.path.join(data_dir, "features.parquet")

    if not os.path.exists(bm25_path) or not os.path.exists(feats_path):
        return None, None, None

    print(f"[rank] Loading BM25 index from {bm25_path}...")
    bm25_model, candidate_ids = load_bm25_artifacts(data_dir)

    print(f"[rank] Loading feature matrix from {feats_path}...")
    features_df = pd.read_parquet(feats_path)

    return bm25_model, candidate_ids, features_df


def score_from_features_df(row: dict, bm25_score: float) -> dict:
    """
    Score a candidate using pre-computed feature row from parquet.
    Much faster than recomputing from raw candidate data.
    """
    # Extract feature groups from prefixed columns
    career_feats = {str(k)[7:]: v for k, v in row.items() if str(k).startswith("career_")}
    tech_feats = {str(k)[5:]: v for k, v in row.items() if str(k).startswith("tech_")}
    behavioral_feats = {str(k)[11:]: v for k, v in row.items() if str(k).startswith("behavioral_")}
    risk_feats = {str(k)[5:]: v for k, v in row.items() if str(k).startswith("risk_")}
    latent_feats = {str(k)[7:]: v for k, v in row.items() if str(k).startswith("latent_")}

    hp_mult = float(row.get("honeypot_multiplier", 1.0))
    st_mult = float(row.get("stuffer_multiplier", 1.0))
    trap_reason = str(row.get("trap_reason", "clean"))

    # Hard-drop definite honeypots AND suspicious candidates
    if hp_mult == 0.0 or hp_mult < SUSPICIOUS_HONEYPOT_THRESHOLD:
        return {
            "score": 0.0, "component_scores": {}, "trap_reasons": trap_reason,
            "is_honeypot": True, "latent_features": latent_feats,
            "career_features": career_feats, "tech_features": tech_feats,
            "behavioral_features": behavioral_feats,
        }

    career_score = compute_career_trajectory_score(career_feats, latent_feats)
    tech_score = compute_technical_fit_score(tech_feats, latent_feats)
    behavioral_score = compute_behavioral_score(behavioral_feats, latent_feats)
    availability_score = compute_availability_score(behavioral_feats, latent_feats)
    education_score = compute_education_score(tech_feats)

    base_score = (
        WEIGHTS["career_trajectory"] * career_score
        + WEIGHTS["technical_fit"] * tech_score
        + WEIGHTS["behavioral"] * behavioral_score
        + WEIGHTS["availability"] * availability_score
        + WEIGHTS["bm25_score"] * bm25_score
        + WEIGHTS["education"] * education_score
    )

    consulting_mult = (
        CONSULTING_ONLY_MULTIPLIER
        if risk_feats.get("is_consulting_only", 0) > 0.5
        else 1.0
    )

    current_title = str(row.get("_current_title", "")).lower()
    title_mult = (
        DISQUALIFYING_TITLE_MULTIPLIER
        if any(dt in current_title for dt in DISQUALIFYING_TITLES)
        else 1.0
    )

    behavioral_gate = float(latent_feats.get("behavioral_gate", 0.5))
    final_score = base_score * behavioral_gate * hp_mult * st_mult * consulting_mult * title_mult
    final_score = max(0.0, min(1.0, final_score))

    return {
        "score": final_score,
        "component_scores": {
            "career": career_score, "technical": tech_score,
            "behavioral": behavioral_score, "availability": availability_score,
            "bm25": bm25_score, "education": education_score,
        },
        "multipliers": {
            "honeypot": hp_mult, "stuffer": st_mult,
            "consulting": consulting_mult, "title": title_mult,
        },
        "trap_reasons": trap_reason,
        "is_honeypot": False,
        "latent_features": latent_feats,
        "career_features": career_feats,
        "tech_features": tech_feats,
        "behavioral_features": behavioral_feats,
    }


def build_minimal_candidate_for_reasoning(row) -> dict:
    """Reconstruct a minimal candidate dict for reasoning generation from feature row."""
    return {
        "candidate_id": row.name,
        "profile": {
            "years_of_experience": row.get("_yoe", 0),
            "current_title": row.get("_current_title", ""),
            "current_company": row.get("_current_company", ""),
            "location": row.get("_location", ""),
        },
        "career_history": [],
        "skills": _reconstruct_skills(str(row.get("_top_skills", ""))),
        "_top_company": str(row.get("_top_company", row.get("_current_company", ""))),
        "redrob_signals": {
            "recruiter_response_rate": row.get("_rrr", 0),
            "notice_period_days": int(row.get("_notice", 60)),
            "open_to_work_flag": bool(row.get("_otw", 0)),
        }
    }


def _reconstruct_skills(skills_str: str) -> list:
    """Reconstruct a minimal skills list from cached top-skills string."""
    if not skills_str or skills_str == "nan":
        return []
    return [
        {"name": s.strip(), "proficiency": "advanced", "endorsements": 10}
        for s in skills_str.split(",") if s.strip()
    ]


def main():
    args = parse_args()
    t_wall = time.time()

    print("=" * 60)
    print("Redrob Intelligent Candidate Ranker")
    print("=" * 60)

    # ── Step 1: Load precomputed artifacts ──────────────────────
    bm25_model, candidate_ids, features_df = load_precomputed(args.data)
    has_precomputed = bm25_model is not None

    model = None
    model_feature_names = None
    if has_precomputed:
        model_path = os.path.join(args.data, "ltr_model.txt")
        feature_names_path = os.path.join(args.data, "feature_names.json")
        if os.path.exists(model_path) and os.path.exists(feature_names_path):
            import lightgbm as lgb
            print(f"[rank] Loading LTR model from {model_path}...")
            model = lgb.Booster(model_file=model_path)
            with open(feature_names_path, "r") as f:
                model_feature_names = json.load(f)

    if not has_precomputed or candidate_ids is None or features_df is None or bm25_model is None:
        has_precomputed = False
        print("[rank] No precomputed artifacts found. Running in slow mode...")
        print("[rank] TIP: Run `python precompute.py` first for much faster ranking.")

        # Load candidates fresh
        if args.sample:
            candidates = load_sample_candidates("sample_candidates.json")
        else:
            candidates = list(tqdm(
                stream_candidates(args.candidates),
                desc="Loading candidates", unit="cand"
            ))
        print(f"[rank] Loaded {len(candidates)} candidates.")

        print("[rank] Building BM25 index...")
        from ranker.bm25_retrieval import build_bm25_index
        bm25_model, candidate_ids = build_bm25_index(candidates, verbose=True)
        candidate_map = {c["candidate_id"]: c for c in candidates}
    else:
        num_feats = len(features_df) if features_df is not None else 0
        print(f"[rank] Precomputed features: {num_feats} candidates")
        candidate_map = None  # will use features_df

    # Assert non-None types to narrow type checker scope
    assert bm25_model is not None
    assert candidate_ids is not None
    assert features_df is not None

    # ── Step 2: BM25 Retrieval (Score all candidates) ────────────────
    top_k = len(candidate_ids)
    print(f"\n[rank] Stage 2: BM25 retrieval (scoring all {top_k} candidates)...")
    t_bm25 = time.time()
    bm25_results = query_bm25(bm25_model, candidate_ids, JD_QUERY, top_k)
    bm25_score_map = dict(bm25_results)
    print(f"[rank] BM25 done in {time.time()-t_bm25:.2f}s — retrieved {len(bm25_results)} candidates")

    # ── Step 3: Score top-K candidates ──────────────────────────
    print(f"\n[rank] Stage 3: Scoring {len(bm25_results)} candidates...")
    t_score = time.time()

    scored_candidates = []

    if has_precomputed:
        if model is not None and model_feature_names is not None:
            # ML Model batch prediction path
            print(f"[rank] Running LTR model batch inference on {len(bm25_results)} candidates...")
            valid_results = [(cid, score) for cid, score in bm25_results if cid in features_df.index]
            cids = [cid for cid, _ in valid_results]
            bm25_scores = [score for _, score in valid_results]
            
            # Sliced feature matrix
            X_df = features_df.loc[cids].copy()
            X_df["bm25_score"] = bm25_scores
            
            # Extract features aligned with model feature list
            X = X_df[model_feature_names].astype(float)
            
            # Predict base scores
            preds = np.asarray(model.predict(X))
            
            # Min-max scale the predictions to [0, 1] to keep them consistent with scoring formula
            min_p = preds.min()
            max_p = preds.max()
            if max_p > min_p:
                norm_preds = (preds - min_p) / (max_p - min_p)
            else:
                norm_preds = np.zeros_like(preds)
                
            # Convert to dictionary for fast O(1) row lookup
            X_df_dict = X_df.to_dict(orient='index')
            for i, cid in enumerate(cids):
                row = X_df_dict[cid]
                hp_mult = float(row.get("honeypot_multiplier", 1.0))
                st_mult = float(row.get("stuffer_multiplier", 1.0))
                trap_reason = str(row.get("trap_reason", "clean"))
                
                # Consulting multiplier
                is_consulting = float(row.get("risk_is_consulting_only", 0.0)) > 0.5
                consulting_mult = CONSULTING_ONLY_MULTIPLIER if is_consulting else 1.0
                
                # Title multiplier
                current_title = str(row.get("_current_title", "")).lower()
                title_mult = (
                    DISQUALIFYING_TITLE_MULTIPLIER
                    if any(dt in current_title for dt in DISQUALIFYING_TITLES)
                    else 1.0
                )
                
                # Hard-drop suspicious honeypots
                is_honeypot = hp_mult == 0.0 or hp_mult < SUSPICIOUS_HONEYPOT_THRESHOLD
                if is_honeypot:
                    final_score = 0.0
                else:
                    behavioral_gate = float(row.get("latent_behavioral_gate", 0.5))
                    final_score = norm_preds[i] * behavioral_gate * hp_mult * st_mult * consulting_mult * title_mult
                    final_score = max(0.0, min(1.0, final_score))
                
                # Extract original features for reasoning
                career_feats = {str(k)[7:]: v for k, v in row.items() if str(k).startswith("career_")}
                tech_feats = {str(k)[5:]: v for k, v in row.items() if str(k).startswith("tech_")}
                behavioral_feats = {str(k)[11:]: v for k, v in row.items() if str(k).startswith("behavioral_")}
                latent_feats = {str(k)[7:]: v for k, v in row.items() if str(k).startswith("latent_")}
                
                scored_candidates.append({
                    "candidate_id": cid,
                    "score": final_score,
                    "is_honeypot": is_honeypot,
                    "trap_reasons": trap_reason,
                    "component_scores": {
                        "career": float(row.get("career_pedigree_score", 0.5)),
                        "technical": float(row.get("tech_core_skill_weighted_score", 0.5)),
                        "behavioral": float(row.get("behavioral_recency_score", 0.5)),
                        "availability": float(row.get("latent_availability_composite", 0.5)),
                        "bm25": bm25_scores[i],
                        "education": float(row.get("tech_education_tier_score", 0.35)),
                    },
                    "career_features": career_feats,
                    "tech_features": tech_feats,
                    "behavioral_features": behavioral_feats,
                    "latent_features": latent_feats,
                })
        else:
            # Fast path (heuristic): use precomputed features
            features_dict = features_df.to_dict(orient='index')
            for cid, bm25_score in tqdm(bm25_results, desc="Scoring"):
                if cid not in features_dict:
                    continue
                row = features_dict[cid]
                result = score_from_features_df(row, bm25_score)
                result["candidate_id"] = cid
                scored_candidates.append(result)
    else:
        # Slow path: compute features on-the-fly
        from ranker.scorer import score_candidate
        assert candidate_map is not None
        for cid, bm25_score in tqdm(bm25_results, desc="Scoring"):
            cand = candidate_map.get(cid)
            if cand is None:
                continue
            result = score_candidate(cand, bm25_score)
            result["candidate_id"] = cid
            result["_candidate"] = cand
            scored_candidates.append(result)

    print(f"[rank] Scoring done in {time.time()-t_score:.2f}s")

    # ── Step 4: Sort and select top-100 ─────────────────────────
    # Filter out honeypots (is_honeypot == True) completely from top ranks to comply with strict evaluation rules
    clean_candidates = [c for c in scored_candidates if not c.get("is_honeypot", False)]
    honeypot_candidates = [c for c in scored_candidates if c.get("is_honeypot", False)]
    
    print(f"\n[rank] Total candidates scored: {len(scored_candidates)}")
    print(f"[rank] Honeypots filtered out from top ranks: {len(honeypot_candidates)}")
    print(f"[rank] Clean candidates available for ranking: {len(clean_candidates)}")

    # Primary: score descending. Tie-break: candidate_id ascending (validator rule)
    clean_candidates.sort(key=lambda x: (-round(x["score"], 4), x["candidate_id"]))
    top_candidates = clean_candidates[:FINAL_TOP_N]

    # Honeypot check (should be exactly 0)
    honeypots_in_top = sum(1 for c in top_candidates if c.get("is_honeypot", False))
    print(f"\n[rank] Top-{FINAL_TOP_N} selected. Honeypots in top: {honeypots_in_top} (limit: 0)")

    # ── Step 5: Generate reasoning ───────────────────────────────
    print("[rank] Generating reasoning strings...")
    t_reason = time.time()

    output_rows = []
    for rank_idx, result in enumerate(top_candidates, start=1):
        result_dict = dict(result) if isinstance(result, dict) else {}
        cid = str(result_dict.get("candidate_id", ""))

        if has_precomputed and cid in features_df.index:
            minimal_cand = build_minimal_candidate_for_reasoning(features_df.loc[cid])
        elif not has_precomputed:
            minimal_cand = result.get("_candidate", {})
        else:
            minimal_cand = {"candidate_id": cid, "profile": {}, "career_history": [],
                            "skills": [], "redrob_signals": {}}

        # Coerce types to dict and float to satisfy static type checkers
        min_cand_dict = dict(minimal_cand) if isinstance(minimal_cand, dict) else {}

        reasoning = generate_reasoning(min_cand_dict, result_dict, rank_idx)
        reasoning = truncate_reasoning(reasoning)

        raw_score = result_dict.get("score", 0.0)
        score_val = float(raw_score) if isinstance(raw_score, (int, float, str)) else 0.0
        output_rows.append({
            "candidate_id": cid,
            "rank": rank_idx,
            "score": round(score_val, 4),
            "reasoning": reasoning,
        })

    print(f"[rank] Reasoning done in {time.time()-t_reason:.2f}s")

    # ── Step 6: Write CSV ────────────────────────────────────────
    print(f"\n[rank] Writing {args.output}...")
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        writer.writerows(output_rows)

    t_total = time.time() - t_wall
    print(f"\n{'='*60}")
    print(f"[rank] DONE! Total time: {t_total:.1f}s ({t_total/60:.2f} min)")
    print(f"[rank] Output: {os.path.abspath(args.output)}")
    print(f"[rank] Top-5 candidates:")
    for row in output_rows[:5]:
        print(f"  #{row['rank']} {row['candidate_id']} score={row['score']:.4f}")
    print(f"  ...")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
