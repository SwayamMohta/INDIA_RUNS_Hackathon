"""
feature_frame.py — Build the candidate feature matrix shared by precompute and rank.

This is the single source of truth for turning a list of raw candidate dicts into
the pandas DataFrame the LightGBM LTR model consumes. It is used in two places:

  * precompute.py — once, offline, over the full 100k pool (saved to features.parquet).
  * rank.py (--standalone / sandbox) — on the fly, over a small sample (<=100), so the
    *trained* model can score candidates that were never precomputed. No training and
    no re-precomputation of the full pool happen at inference time.

Keeping this logic in one function guarantees the on-the-fly features are byte-for-byte
identical in construction to the precomputed ones, so the sandbox demonstrates the same
system that produced the submission.
"""

from typing import Any, Dict, List

import pandas as pd
from tqdm import tqdm

from ranker.features import compute_all_features, _classify_company
from ranker.honeypot import get_trap_multipliers


def build_feature_row(cand: Dict[str, Any]) -> Dict[str, Any]:
    """Compute the full feature row (features + trap multipliers + cached raw fields)
    for a single candidate. Mirrors exactly what precompute.py persists per candidate."""
    cid = cand.get("candidate_id", "")
    feats = compute_all_features(cand)
    hp_mult, st_mult, trap_reason = get_trap_multipliers(cand)
    feats["candidate_id"] = cid
    feats["honeypot_multiplier"] = hp_mult
    feats["stuffer_multiplier"] = st_mult
    feats["trap_reason"] = trap_reason

    # Raw fields cached for reasoning generation (prefixed with "_" to keep them out
    # of the model feature space, which is selected by career_/tech_/... prefixes).
    profile = cand.get("profile", {})
    feats["_yoe"] = profile.get("years_of_experience", 0)
    feats["_current_title"] = profile.get("current_title", "")
    feats["_current_company"] = profile.get("current_company", "")
    feats["_location"] = profile.get("location", "")
    signals = cand.get("redrob_signals", {})
    feats["_rrr"] = signals.get("recruiter_response_rate", 0)
    feats["_notice"] = signals.get("notice_period_days", 60)
    feats["_otw"] = int(signals.get("open_to_work_flag", False))

    # Cache top JD-relevant skills for reasoning generation.
    skills = cand.get("skills", [])
    top_skills = [
        s.get("name", "") for s in skills
        if s.get("proficiency") in ("expert", "advanced") and s.get("endorsements", 0) > 5
    ][:4]
    feats["_top_skills"] = ", ".join(top_skills) if top_skills else ""

    # Cache the most senior product-company name for reasoning generation.
    career = cand.get("career_history", [])
    product_cos = [
        r.get("company", "") for r in career
        if _classify_company(r.get("company", "")) in ("tier1", "tier2", "tier3")
        and r.get("company", "") not in ("", "Unknown")
    ]
    feats["_top_company"] = product_cos[0] if product_cos else profile.get("current_company", "")

    return feats


def build_feature_frame(
    candidates: List[Dict[str, Any]],
    show_progress: bool = False,
) -> pd.DataFrame:
    """Build a feature DataFrame (indexed by candidate_id) from raw candidate dicts.

    Candidates that fail feature computation are kept with an ``error`` column so the
    index still aligns with the input list (matching precompute.py's behaviour).
    """
    rows: List[Dict[str, Any]] = []
    failed = 0
    iterator = (
        tqdm(candidates, desc="Computing features", unit="cand")
        if show_progress else candidates
    )
    for cand in iterator:
        cid = cand.get("candidate_id", "")
        try:
            rows.append(build_feature_row(cand))
        except Exception as e:  # noqa: BLE001 — never let one bad record abort the batch
            failed += 1
            if failed <= 10:
                print(f"[feature_frame] WARNING: failed on {cid}: {e}")
            rows.append({"candidate_id": cid, "error": str(e)})

    df = pd.DataFrame(rows).set_index("candidate_id")
    return df
