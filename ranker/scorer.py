"""
scorer.py — The final scoring formula.

Combines all feature groups into a final rank score using a weighted formula.
Applies multiplicative penalties for honeypots, keyword stuffers,
and consulting-only profiles.

Final Score formula:
  base_score = (
      0.35 * career_trajectory_score
    + 0.25 * technical_fit_score
    + 0.15 * behavioral_score
    + 0.10 * availability_score
    + 0.10 * bm25_score_normalized
    + 0.05 * education_score
  )
  final_score = base_score * honeypot_mult * stuffer_mult * consulting_mult
"""

from typing import Dict, Any, Optional

from ranker.config import (
    WEIGHTS,
    HONEYPOT_MULTIPLIER, KEYWORD_STUFFER_MULTIPLIER,
    CONSULTING_ONLY_MULTIPLIER, DISQUALIFYING_TITLE_MULTIPLIER,
    CV_MISMATCH_FLOOR, CV_MISMATCH_CEIL, CV_MISMATCH_TRIGGER,
    TITLE_CHASER_MULTIPLIER,
)
from ranker.honeypot import get_trap_multipliers
from ranker.features import (
    compute_career_features, compute_technical_features,
    compute_behavioral_features, compute_risk_features,
    compute_latent_features,
)


# ─────────────────────────────────────────────────────────────────
# GRADED PENALTY MULTIPLIERS
# Shared by both the LTR path (rank.py) and the heuristic path so they
# stay consistent. Each returns a value in (0, 1].
# ─────────────────────────────────────────────────────────────────

def compute_cv_mismatch_multiplier(domain_mismatch_score: float) -> float:
    """Graded penalty for CV/speech/robotics-primary profiles with thin NLP/IR.

    Returns 1.0 below the trigger, interpolating down to CV_MISMATCH_FLOOR at a
    fully off-domain profile (score = 1.0). Encodes the JD's explicit reject of
    'primary expertise is computer vision, speech, or robotics without significant
    NLP/IR exposure'.
    """
    s = float(domain_mismatch_score)
    if s <= CV_MISMATCH_TRIGGER:
        return CV_MISMATCH_CEIL
    frac = (s - CV_MISMATCH_TRIGGER) / (1.0 - CV_MISMATCH_TRIGGER)
    return CV_MISMATCH_CEIL - frac * (CV_MISMATCH_CEIL - CV_MISMATCH_FLOOR)


def compute_consulting_multiplier(service_ratio: float, is_consulting_only: float = 0.0) -> float:
    """Graded consulting penalty scaled by service-firm tenure share.

    Full product-company career (service_ratio=0) → 1.0; all-service → the
    CONSULTING_ONLY_MULTIPLIER floor. Only penalizes when service work dominates
    (>50% of tenure), so a single short stint at a service firm is not punished.
    """
    sr = float(service_ratio)
    if is_consulting_only > 0.5:
        return CONSULTING_ONLY_MULTIPLIER
    if sr <= 0.5:
        return 1.0
    frac = (sr - 0.5) / 0.5
    return 1.0 - frac * (1.0 - CONSULTING_ONLY_MULTIPLIER)


def compute_title_chaser_multiplier(is_title_chaser: float) -> float:
    """Mild penalty for serial short-tenure title-hopping."""
    return TITLE_CHASER_MULTIPLIER if float(is_title_chaser) > 0.5 else 1.0


def compute_career_trajectory_score(career_feats: Dict, latent_feats: Dict) -> float:
    """
    Career trajectory composite score (weight: 0.38).
    Rewards product company experience, career progression, and shipping history.
    """
    score = (
        0.25 * career_feats.get("product_ratio", 0)
        + 0.20 * career_feats.get("pedigree_score", 0)
        + 0.20 * career_feats.get("domain_shipping_score", 0)
        + 0.15 * career_feats.get("title_alignment_score", 0)
        + 0.10 * career_feats.get("experience_range_fit", 0)
        + 0.05 * career_feats.get("shipper_verb_score", 0)
        + 0.05 * career_feats.get("career_progression_score", 0)
    )
    # Bonus if currently at product company
    if career_feats.get("current_company_tier_score", 0) >= 0.8:
        score = min(1.0, score * 1.1)
    return score


def compute_technical_fit_score(tech_feats: Dict, latent_feats: Dict) -> float:
    """
    Technical fit composite score (weight: 0.25).
    Rewards deep alignment with JD's technical requirements.
    """
    score = (
        0.30 * tech_feats.get("core_skill_weighted_score", 0)
        + 0.20 * latent_feats.get("domain_specialist_score", 0)
        + 0.15 * tech_feats.get("description_technical_depth", 0)
        + 0.15 * latent_feats.get("technical_depth_score", 0)
        + 0.10 * tech_feats.get("github_score", 0)
        + 0.10 * tech_feats.get("relevant_assessment_score", 0)
    )
    return score


def compute_behavioral_score(behavioral_feats: Dict, latent_feats: Dict) -> float:
    """
    Soft behavioral composite score (weight: 0.08).
    Excludes core 3 signals now in gate.
    """
    score = (
        0.35 * latent_feats.get("engagement_composite", 0)
        + 0.30 * behavioral_feats.get("interview_completion_rate", 0)
        + 0.20 * behavioral_feats.get("verification_score", 0)
        + 0.15 * behavioral_feats.get("profile_completeness", 0)
    )
    return score


def compute_availability_score(behavioral_feats: Dict, latent_feats: Dict) -> float:
    """
    Availability composite score (weight: 0.10).
    Rewards candidates who can start soon and are in target location.
    """
    return latent_feats.get("availability_composite", 0)


def compute_education_score(tech_feats: Dict) -> float:
    """Education composite score (weight: 0.05)."""
    score = (
        0.70 * tech_feats.get("education_tier_score", 0.35)
        + 0.30 * tech_feats.get("cs_degree_flag", 0)
    )
    return score


def score_candidate(
    candidate: Dict[str, Any],
    bm25_score: float = 0.0,
    precomputed_features: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Compute the final score for a single candidate.

    Args:
        candidate: Raw candidate dict
        bm25_score: Normalized BM25 score (0-1) from Stage 2 retrieval
        precomputed_features: Pre-computed feature dict (optional, for speed)

    Returns:
        Dict with 'score', 'component_scores', 'trap_reasons', 'features'
    """
    # ── Trap Detection ──────────────────────────────────────────
    hp_mult, st_mult, trap_reason = get_trap_multipliers(candidate)

    # Short-circuit for definite honeypots
    if hp_mult == 0.0:
        return {
            "score": 0.0,
            "component_scores": {},
            "trap_reasons": trap_reason,
            "is_honeypot": True,
        }

    # ── Feature Computation ──────────────────────────────────────
    if precomputed_features:
        # Use pre-computed features if available (fast path)
        feats = precomputed_features
        career_feats = {str(k)[7:]: v for k, v in feats.items() if str(k).startswith("career_")}
        tech_feats = {str(k)[5:]: v for k, v in feats.items() if str(k).startswith("tech_")}
        behavioral_feats = {str(k)[11:]: v for k, v in feats.items() if str(k).startswith("behavioral_")}
        risk_feats = {str(k)[5:]: v for k, v in feats.items() if str(k).startswith("risk_")}
        latent_feats = {str(k)[7:]: v for k, v in feats.items() if str(k).startswith("latent_")}
    else:
        career_feats = compute_career_features(candidate)
        tech_feats = compute_technical_features(candidate)
        behavioral_feats = compute_behavioral_features(candidate)
        risk_feats = compute_risk_features(candidate)
        latent_feats = compute_latent_features(
            career_feats, tech_feats, behavioral_feats, risk_feats
        )

    # ── Component Scores ─────────────────────────────────────────
    career_score = compute_career_trajectory_score(career_feats, latent_feats)
    tech_score = compute_technical_fit_score(tech_feats, latent_feats)
    behavioral_score = compute_behavioral_score(behavioral_feats, latent_feats)
    availability_score = compute_availability_score(behavioral_feats, latent_feats)
    education_score = compute_education_score(tech_feats)

    # ── Base Score ───────────────────────────────────────────────
    base_score = (
        WEIGHTS["career_trajectory"] * career_score
        + WEIGHTS["technical_fit"] * tech_score
        + WEIGHTS["behavioral"] * behavioral_score
        + WEIGHTS["availability"] * availability_score
        + WEIGHTS["bm25_score"] * bm25_score
        + WEIGHTS["education"] * education_score
    )

    # ── Graded Consulting Multiplier ─────────────────────────────
    consulting_mult = compute_consulting_multiplier(
        risk_feats.get("service_ratio", 0.0),
        risk_feats.get("is_consulting_only", 0.0),
    )

    # ── Domain-Mismatch (CV/speech/robotics) Multiplier ──────────
    cv_mismatch_mult = compute_cv_mismatch_multiplier(
        risk_feats.get("domain_mismatch_score", 0.0)
    )

    # ── Title-Chaser Multiplier ──────────────────────────────────
    title_chaser_mult = compute_title_chaser_multiplier(
        risk_feats.get("is_title_chaser", 0.0)
    )

    # ── Disqualifying Title Multiplier ───────────────────────────
    from ranker.config import DISQUALIFYING_TITLES
    profile = candidate.get("profile", {})
    current_title = profile.get("current_title", "").lower()
    title_mult = (
        DISQUALIFYING_TITLE_MULTIPLIER
        if any(dt in current_title for dt in DISQUALIFYING_TITLES)
        else 1.0
    )

    # ── Final Score ──────────────────────────────────────────────
    behavioral_gate = latent_feats.get("behavioral_gate", 0.5)
    final_score = (
        base_score * behavioral_gate * hp_mult * st_mult
        * consulting_mult * cv_mismatch_mult * title_chaser_mult * title_mult
    )
    final_score = max(0.0, min(1.0, final_score))

    return {
        "score": final_score,
        "component_scores": {
            "career": career_score,
            "technical": tech_score,
            "behavioral": behavioral_score,
            "availability": availability_score,
            "bm25": bm25_score,
            "education": education_score,
        },
        "multipliers": {
            "honeypot": hp_mult,
            "stuffer": st_mult,
            "consulting": consulting_mult,
            "cv_mismatch": cv_mismatch_mult,
            "title_chaser": title_chaser_mult,
            "title": title_mult,
        },
        "trap_reasons": trap_reason,
        "is_honeypot": hp_mult == 0.0,
        "latent_features": latent_feats,
        "career_features": career_feats,
        "tech_features": tech_feats,
        "behavioral_features": behavioral_feats,
    }
