"""
train_ltr.py — Train LightGBM LambdaMART ranker with improved methodology.

Key improvements over previous version:
  1. Multi-signal synthetic labels (not circular from heuristic scorer)
  2. Training on ALL candidates (not just top-10K BM25)
  3. Multiple query groups for proper listwise training
  4. Validation split with NDCG evaluation
  5. New features (interaction_, additional_) included automatically
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import lightgbm as lgb

# Add project root to path (where ranker/ is located)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ranker.config import (
    JD_QUERY, WEIGHTS, ALIGNED_TITLES, DISQUALIFYING_TITLES,
    TIER_1_COMPANIES, TIER_2_COMPANIES, TIER_3_COMPANIES, SERVICE_FIRMS,
    CORE_SKILLS, SUSPICIOUS_HONEYPOT_THRESHOLD,
)
from ranker.bm25_retrieval import load_bm25_artifacts, query_bm25


def compute_synthetic_relevance(row, bm25_score):
    """
    Compute synthetic relevance labels using the new multi-signal rules:
      3 = Highly relevant: product_ratio > 0.7, domain_shipping_score > 0.5,
          technical_fit strong (>0.6), behavioral_gate > 0.6.
      2 = Relevant: product company experience (product_ratio > 0.4),
          adjacent technical fit (>0.4), acceptable hireability (behavioral_gate > 0.4).
      1 = Marginally relevant: some useful signal but missing key dimensions.
      0 = Irrelevant: consulting-only, disqualifying title, or definitively inactive/honeypot.
    """
    # ── Relevance 0 check ──────────────────────────────────────────
    current_title = str(row.get("_current_title", "")).lower()
    is_disqualified = any(dt in current_title for dt in DISQUALIFYING_TITLES)
    is_consulting = float(row.get("risk_is_consulting_only", 0.0)) > 0.5
    hp_mult = float(row.get("honeypot_multiplier", 1.0))
    is_honeypot = hp_mult == 0.0 or hp_mult < SUSPICIOUS_HONEYPOT_THRESHOLD
    is_inactive = float(row.get("behavioral_recency_score", 0.0)) < 0.05
    # JD explicitly rejects CV/speech/robotics-primary profiles with thin NLP/IR.
    is_off_domain = float(row.get("risk_domain_mismatch_score", 0.0)) > 0.65

    if is_disqualified or is_consulting or is_honeypot or is_inactive or is_off_domain:
        return 0

    # ── Compute Technical Fit Score ──────────────────────────────
    core_skill_score = float(row.get("tech_core_skill_weighted_score", 0))
    domain_specialist = float(row.get("latent_domain_specialist_score", 0))
    desc_depth = float(row.get("tech_description_technical_depth", 0))
    technical_depth = float(row.get("latent_technical_depth_score", 0))
    github = float(row.get("tech_github_score", 0.45))
    assessment = float(row.get("tech_relevant_assessment_score", 0))
    
    technical_fit_score = (
        0.30 * core_skill_score
        + 0.20 * domain_specialist
        + 0.15 * desc_depth
        + 0.15 * technical_depth
        + 0.10 * github
        + 0.10 * assessment
    )

    # ── Retrieve key comparison values ────────────────────────────
    product_ratio = float(row.get("career_product_ratio", 0.0))
    domain_shipping = float(row.get("career_domain_shipping_score", 0.0))
    behavioral_gate = float(row.get("latent_behavioral_gate", 0.5))

    # ── Tier Assignment ───────────────────────────────────────────
    if (product_ratio > 0.7 and 
        domain_shipping > 0.5 and 
        technical_fit_score > 0.6 and 
        behavioral_gate > 0.6):
        return 3
    elif (product_ratio > 0.4 and 
          technical_fit_score > 0.4 and 
          behavioral_gate > 0.4):
        return 2
    else:
        return 1


def main():
    print("=" * 60)
    print("Training LightGBM LambdaMART Ranker (Improved)")
    print("=" * 60)

    # Data directory: --data <dir>, else default to project_root/input.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, "input")
    if "--data" in sys.argv:
        data_dir = sys.argv[sys.argv.index("--data") + 1]
    
    # ── Step 1: Load precomputed data ────────────────────────────
    bm25_path = os.path.join(data_dir, "bm25_index.pkl")
    feats_path = os.path.join(data_dir, "features.parquet")
    
    if not os.path.exists(bm25_path) or not os.path.exists(feats_path):
        print(f"Error: Precomputed artifacts not found in {data_dir}. Run precompute.py first.")
        sys.exit(1)
        
    print(f"Loading BM25 index from {bm25_path}...")
    bm25_model, candidate_ids = load_bm25_artifacts(data_dir)
    
    print(f"Loading feature matrix from {feats_path}...")
    features_df = pd.read_parquet(feats_path)
    print(f"  Feature matrix: {features_df.shape[0]} candidates, {features_df.shape[1]} columns")
    
    # ── Step 2: BM25 score ALL candidates ────────────────────────
    print(f"Running BM25 retrieval for ALL {len(candidate_ids)} candidates...")
    bm25_results = query_bm25(bm25_model, candidate_ids, JD_QUERY, len(candidate_ids))
    bm25_score_map = dict(bm25_results)
    
    # ── Step 3: Compute synthetic relevance labels ───────────────
    print("Computing multi-signal synthetic relevance labels...")
    
    valid_ids = [cid for cid in candidate_ids if cid in features_df.index]
    print(f"  {len(valid_ids)} candidates in feature matrix")
    
    labels = []
    for cid in valid_ids:
        row = features_df.loc[cid]
        bm25_score = bm25_score_map.get(cid, 0.0)
        label = compute_synthetic_relevance(row, bm25_score)
        labels.append(label)
    
    labels = np.array(labels, dtype=int)
    
    # Print label distribution
    print("  Label distribution:")
    for tier in range(5):
        count = (labels == tier).sum()
        pct = count / len(labels) * 100
        print(f"    Tier {tier}: {count:>6} ({pct:.1f}%)")
    
    # ── Step 4: Build training data ──────────────────────────────
    print("\nBuilding training feature matrix...")
    
    train_df = features_df.loc[valid_ids].copy()
    train_df["bm25_score"] = [bm25_score_map.get(cid, 0.0) for cid in valid_ids]
    
    # Select all numeric feature columns (auto-discover new features)
    feature_cols = [
        col for col in train_df.columns
        if col.startswith(("career_", "tech_", "behavioral_", "risk_", "latent_",
                           "interaction_", "additional_"))
    ]
    feature_cols.append("bm25_score")
    
    # Filter to only numeric columns
    feature_cols = [col for col in feature_cols if train_df[col].dtype in ['float64', 'float32', 'int64', 'int32']]
    
    print(f"  Selected {len(feature_cols)} features for training")
    
    X = train_df[feature_cols].astype(float).values
    y = labels
    
    # ── Step 5: Few large representative query groups ────────────
    # This is a single-query ranking problem (every candidate is scored against the
    # SAME JD). The original used random 500-item groups, which only let LambdaMART
    # compare candidates within tiny subsets — discarding most ordering signal and
    # injecting noise. The ideal is one group over the whole list, but LightGBM caps
    # a query group at 10,000 rows, so we use a few LARGE groups (~8k) instead. After
    # a global shuffle each group is a representative random sample of the full tier
    # mix, so within-group pairwise comparisons approximate the global ranking — far
    # more comparisons per group than 500, while respecting LightGBM's per-query cap.
    n_total = len(X)
    GROUP_SIZE = 8000  # <= LightGBM's 10,000 per-query limit

    # Shuffle once so every group (and the held-out val slice) is representative.
    np.random.seed(42)
    shuffle_idx = np.random.permutation(n_total)
    X = X[shuffle_idx]
    y = y[shuffle_idx]

    n_groups = max(1, n_total // GROUP_SIZE)
    groups = [GROUP_SIZE] * n_groups
    remainder = n_total - GROUP_SIZE * n_groups
    if remainder > 0:
        groups.append(remainder)
    print(f"  Created {len(groups)} query groups (size ~{GROUP_SIZE})")

    # ── Step 6: Train/Val split (whole groups → validation) ──────
    n_val_groups = max(1, len(groups) // 5)
    n_train_groups = len(groups) - n_val_groups
    train_end = sum(groups[:n_train_groups])

    X_train, X_val = X[:train_end], X[train_end:]
    y_train, y_val = y[:train_end], y[train_end:]
    groups_train = groups[:n_train_groups]
    groups_val = groups[n_train_groups:]

    print(f"  Train: {len(X_train)} candidates in {len(groups_train)} groups")
    print(f"  Val:   {len(X_val)} candidates in {len(groups_val)} groups")
    
    # ── Step 7: Train LGBMRanker model ───────────────────────────
    print("\nTraining LGBMRanker LambdaMART model...")
    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=200,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=5,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        label_gain=[0, 1, 3, 7],
        eval_at=[10, 50],
        verbose=-1,
        n_jobs=1,
    )
    
    ranker.fit(
        X_train, y_train,
        group=groups_train,
        eval_set=[(X_val, y_val)],
        eval_group=[groups_val],
        callbacks=[lgb.log_evaluation(period=50)],
    )
    
    print("Model training complete.")
    
    # ── Step 8: Save artifacts ───────────────────────────────────
    model_path = os.path.join(data_dir, "ltr_model.txt")
    ranker.booster_.save_model(model_path)
    print(f"Saved model to {model_path}")
    
    feature_names_path = os.path.join(data_dir, "feature_names.json")
    with open(feature_names_path, "w") as f:
        json.dump(feature_cols, f)
    print(f"Saved feature list ({len(feature_cols)} features) to {feature_names_path}")
    
    # ── Step 9: Feature Importances ──────────────────────────────
    importances = ranker.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    
    print("\nTop 20 Feature Importances:")
    for idx in sorted_idx[:20]:
        print(f"  {feature_cols[idx]}: {importances[idx]}")
    
    # ── Step 10: Quick validation stats ──────────────────────────
    print("\n" + "=" * 60)
    val_preds = np.asarray(ranker.predict(X_val))
    print(f"Validation predictions - min: {val_preds.min():.4f}, max: {val_preds.max():.4f}, "
          f"mean: {val_preds.mean():.4f}")
    
    # Check correlation between predictions and labels
    from scipy.stats import spearmanr
    corr, pval = spearmanr(val_preds, y_val)
    print(f"Spearman correlation (pred vs label): {corr:.4f} (p={pval:.2e})")
    print("=" * 60)


if __name__ == "__main__":
    main()
