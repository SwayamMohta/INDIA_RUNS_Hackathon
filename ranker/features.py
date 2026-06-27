"""
features.py — Comprehensive feature computation for all 80+ features.

Features are grouped into 5 categories:
  A. Career & Trajectory Features
  B. Technical Fit Features (JD skill alignment)
  C. Behavioral & Availability Features
  D. Risk & Consistency Features
  E. Latent / Derived Features

All features output values in [0, 1] unless otherwise noted.
"""

import math
import re
from datetime import date, datetime
from typing import Dict, Any, List, Optional

from ranker.config import (
    EVAL_DATE, CORE_SKILLS, SHIPPER_VERBS,
    RETRIEVAL_KEYWORDS, RANKING_KEYWORDS, LLM_KEYWORDS, PRODUCTION_KEYWORDS,
    TIER_1_COMPANIES, TIER_2_COMPANIES, TIER_3_COMPANIES, SERVICE_FIRMS,
    DISQUALIFYING_TITLES, ALIGNED_TITLES,
    PREFERRED_LOCATIONS, INDIA_LOCATIONS,
    INACTIVITY_HALF_LIFE_DAYS, MAX_NOTICE_PERIOD_DAYS, IDEAL_NOTICE_PERIOD_DAYS,
    IDEAL_YOE_MIN, IDEAL_YOE_MAX, HARD_MIN_YOE,
)


# ─────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────

def _normalize(value: float, min_val: float, max_val: float) -> float:
    """Clip and normalize value to [0, 1]."""
    if max_val == min_val:
        return 0.0
    return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))


def _days_ago(date_str: Optional[str]) -> Optional[int]:
    """Return number of days since given date string (YYYY-MM-DD)."""
    if not date_str:
        return None
    try:
        d = datetime.strptime(date_str[:10], "%Y-%m-%d").date()
        return (EVAL_DATE - d).days
    except ValueError:
        return None


def _parse_date(date_str: Optional[str]) -> Optional[date]:
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _keyword_score(text: str, keywords: List[str]) -> float:
    """Count unique keyword hits in text (case-insensitive). Returns raw count."""
    text_lower = text.lower()
    hits = sum(1 for kw in keywords if kw in text_lower)
    return hits


def _classify_company(company_name: str) -> str:
    """Classify a company name into tier: tier1, tier2, tier3, service, unknown."""
    name = company_name.lower().strip()
    # Check exact and partial matches
    for known in TIER_1_COMPANIES:
        if known in name:
            return "tier1"
    for known in TIER_2_COMPANIES:
        if known in name:
            return "tier2"
    for known in SERVICE_FIRMS:
        if known in name:
            return "service"
    for known in TIER_3_COMPANIES:
        if known in name:
            return "tier3"
    # Infer from company size + industry
    return "unknown"


COMPANY_TIER_SCORE = {
    "tier1": 1.0,
    "tier2": 0.85,
    "tier3": 0.65,
    "unknown": 0.55,
    "service": 0.25,
}


# ─────────────────────────────────────────────────────────────────
# GROUP A: CAREER & TRAJECTORY FEATURES
# ─────────────────────────────────────────────────────────────────

def compute_career_features(candidate: Dict[str, Any]) -> Dict[str, float]:
    """Compute all career trajectory features."""
    features = {}
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])

    yoe = profile.get("years_of_experience", 0)
    features["total_experience_years"] = yoe

    # Experience range fit (penalize too junior or too senior for JD)
    if yoe < HARD_MIN_YOE:
        features["experience_range_fit"] = 0.1
    elif yoe < IDEAL_YOE_MIN:
        features["experience_range_fit"] = _normalize(yoe, HARD_MIN_YOE, IDEAL_YOE_MIN) * 0.7
    elif yoe <= IDEAL_YOE_MAX:
        features["experience_range_fit"] = 1.0
    else:
        # Too senior can still be ok, just mild penalty
        features["experience_range_fit"] = max(0.6, 1.0 - (yoe - IDEAL_YOE_MAX) * 0.03)

    # Product vs service company breakdown
    product_months = 0
    service_months = 0
    tier_scores = []
    all_service = True
    has_product = False

    for role in career:
        company = role.get("company", "")
        duration = role.get("duration_months", 0)
        industry = role.get("industry", "")
        tier = _classify_company(company)

        tier_scores.append((tier, duration))

        if tier in ("tier1", "tier2", "tier3", "unknown"):
            product_months += duration
            # Check industry as extra signal
            if any(svc in industry.lower() for svc in ["it services", "consulting", "staffing"]):
                # Unknown company in IT services → treat as service
                service_months += duration
                product_months -= duration
            else:
                has_product = True
                all_service = False
        elif tier == "service":
            service_months += duration
        else:
            product_months += duration

    total_career_months = max(1, product_months + service_months)
    features["product_company_years"] = product_months / 12.0
    features["service_company_years"] = service_months / 12.0
    features["product_ratio"] = product_months / total_career_months
    features["is_all_service_firm"] = float(all_service)
    features["has_any_product_company"] = float(has_product)

    # Pedigree score: weighted average of company tier scores by tenure
    if tier_scores:
        weighted_sum = sum(COMPANY_TIER_SCORE[t] * d for t, d in tier_scores)
        total_d = sum(d for _, d in tier_scores)
        features["pedigree_score"] = weighted_sum / max(1, total_d)
    else:
        features["pedigree_score"] = 0.4

    # Current company
    current_company = profile.get("current_company", "")
    current_tier = _classify_company(current_company)
    features["current_company_tier_score"] = COMPANY_TIER_SCORE.get(current_tier, 0.5)

    # Current title alignment
    current_title = profile.get("current_title", "").lower()
    if any(at in current_title for at in ALIGNED_TITLES):
        features["title_alignment_score"] = 1.0
    elif any(dt in current_title for dt in DISQUALIFYING_TITLES):
        features["title_alignment_score"] = 0.1
    else:
        features["title_alignment_score"] = 0.5

    # Career progression: check if titles trend upward
    features["career_progression_score"] = _compute_career_progression(career)

    # Shipper verb count in career descriptions
    all_descriptions = " ".join(r.get("description", "") for r in career).lower()
    shipper_hits = sum(all_descriptions.count(verb) for verb in SHIPPER_VERBS)
    features["shipper_verb_score"] = min(1.0, shipper_hits / 15.0)  # cap at 15 hits = 1.0

    # Career stability: longest tenure
    if career:
        longest = max(r.get("duration_months", 0) for r in career)
        features["longest_tenure_score"] = min(1.0, longest / 36.0)  # 36+ months = 1.0
    else:
        features["longest_tenure_score"] = 0.0

    # Number of roles (not too many, not too few)
    n_roles = len(career)
    if n_roles == 0:
        features["roles_count_score"] = 0.0
    elif n_roles <= 5:
        features["roles_count_score"] = 0.8
    else:
        features["roles_count_score"] = max(0.5, 1.0 - (n_roles - 5) * 0.1)

    # Domain-specific shipping score
    retrieval_shipping = 0.0
    ranking_shipping = 0.0
    recsys_shipping = 0.0
    
    for role in career:
        desc = role.get("description", "").lower()
        if (any(v in desc for v in ["built", "deployed", "shipped", "designed", "engineered"]) and
                any(t in desc for t in ["retrieval", "search", "faiss", "embedding", "vector"])):
            retrieval_shipping = 1.0
        if (any(v in desc for v in ["built", "deployed", "shipped"]) and
                any(t in desc for t in ["ranking", "ltr", "lambdamart", "reranking", "recommendation"])):
            ranking_shipping = 1.0
        if (any(v in desc for v in ["built", "deployed", "shipped"]) and
                any(t in desc for t in ["recommendation", "recommender", "matching"])):
            recsys_shipping = 1.0
            
    features["domain_shipping_score"] = max(
        0.0, min(1.0, (retrieval_shipping + ranking_shipping + recsys_shipping) / 3.0)
    )

    return features


def _compute_career_progression(career: List[Dict]) -> float:
    """
    Detect upward career progression (junior → mid → senior → lead → staff/principal).
    Returns score in [0, 1].
    """
    SENIORITY_MAP = {
        "intern": 0, "trainee": 0, "fresher": 0,
        "junior": 1, "associate": 1,
        "": 2, "engineer": 2, "analyst": 2, "scientist": 2,
        "senior": 3, "sr.": 3, "sr ": 3,
        "lead": 4, "tech lead": 4, "staff": 4,
        "principal": 5, "director": 5, "head": 5, "vp": 6, "manager": 3,
    }
    seniority_levels = []
    for role in career:
        title = role.get("title", "").lower()
        level = 2  # default: mid-level
        for keyword, lvl in sorted(SENIORITY_MAP.items(), key=lambda x: -len(x[0])):
            if keyword and keyword in title:
                level = lvl
                break
        seniority_levels.append(level)

    if len(seniority_levels) < 2:
        return 0.5

    # Check if generally trending upward
    improvements = sum(
        1 for i in range(1, len(seniority_levels))
        if seniority_levels[i] >= seniority_levels[i - 1]
    )
    return improvements / (len(seniority_levels) - 1)


# ─────────────────────────────────────────────────────────────────
# GROUP B: TECHNICAL FIT FEATURES (JD Skill Alignment)
# ─────────────────────────────────────────────────────────────────

def compute_technical_features(candidate: Dict[str, Any]) -> Dict[str, float]:
    """Compute all technical skill alignment features."""
    features = {}
    skills = candidate.get("skills", [])
    career = candidate.get("career_history", [])
    signals = candidate.get("redrob_signals", {})

    # Aggregate text sources
    all_descriptions = " ".join(r.get("description", "") for r in career).lower()
    all_titles = " ".join(r.get("title", "") for r in career).lower()
    skill_names_lower = " ".join(s.get("name", "") for s in skills).lower()
    all_text = f"{all_descriptions} {all_titles} {skill_names_lower}"

    # Core skill hits — weighted by proficiency in skills list
    core_score = 0.0
    core_hit_count = 0
    proficiency_weights = {"expert": 1.0, "advanced": 0.85, "intermediate": 0.6, "beginner": 0.3}

    for skill in skills:
        sname = skill.get("name", "").lower()
        proficiency = skill.get("proficiency", "intermediate")
        weight = proficiency_weights.get(proficiency, 0.5)
        for kw, kw_weight in CORE_SKILLS.items():
            if kw in sname:
                core_score += weight * min(kw_weight, 2.5) / 2.5  # normalize kw_weight
                core_hit_count += 1
                break

    features["core_skill_count"] = float(core_hit_count)
    features["core_skill_weighted_score"] = min(1.0, core_score / 8.0)  # 8+ = max

    # Domain-specific expertise scores
    features["retrieval_score"] = min(1.0, _keyword_score(all_text, RETRIEVAL_KEYWORDS) / 4.0)
    features["ranking_score"] = min(1.0, _keyword_score(all_text, RANKING_KEYWORDS) / 3.0)
    features["llm_score"] = min(1.0, _keyword_score(all_text, LLM_KEYWORDS) / 4.0)
    features["production_ml_score"] = min(1.0, _keyword_score(all_text, PRODUCTION_KEYWORDS) / 4.0)

    # Career description technical depth (vs only skill list hits)
    features["description_technical_depth"] = min(
        1.0,
        _keyword_score(all_descriptions, list(CORE_SKILLS.keys())) / 10.0
    )

    # Skill assessment scores from Redrob platform (verified skills)
    assessment_scores = signals.get("skill_assessment_scores", {})
    if assessment_scores:
        avg_assessment = sum(assessment_scores.values()) / len(assessment_scores)
        # Check if any assessed skills are JD-relevant
        relevant_assessments = [
            v for k, v in assessment_scores.items()
            if any(kw in k.lower() for kw in CORE_SKILLS)
        ]
        if relevant_assessments:
            features["relevant_assessment_score"] = sum(relevant_assessments) / (
                len(relevant_assessments) * 100.0
            )
        else:
            features["relevant_assessment_score"] = 0.0
        features["avg_assessment_score"] = avg_assessment / 100.0
    else:
        features["relevant_assessment_score"] = 0.0
        features["avg_assessment_score"] = 0.0

    # Skill endorsements — normalized (proxy for social proof)
    if skills:
        total_endorsements = sum(s.get("endorsements", 0) for s in skills)
        features["skill_endorsement_score"] = min(1.0, total_endorsements / 200.0)
    else:
        features["skill_endorsement_score"] = 0.0

    # GitHub activity (verified coding evidence)
    github_score = signals.get("github_activity_score", -1)
    if github_score == -1:
        features["github_score"] = 0.45  # no github linked: near-neutral
    else:
        features["github_score"] = github_score / 100.0

    # Education tier
    education = candidate.get("education", [])
    edu_tier_scores = []
    for edu in education:
        t = edu.get("tier", "unknown")
        if t == "tier_1":
            edu_tier_scores.append(1.0)
        elif t == "tier_2":
            edu_tier_scores.append(0.75)
        elif t == "tier_3":
            edu_tier_scores.append(0.5)
        else:
            edu_tier_scores.append(0.35)
    features["education_tier_score"] = max(edu_tier_scores) if edu_tier_scores else 0.35

    # CS/ML relevant field of study
    cs_fields = {"computer science", "computer engineering", "ml", "ai", "data science",
                 "statistics", "mathematics", "electrical engineering", "information technology"}
    has_cs_degree = any(
        any(f in edu.get("field_of_study", "").lower() for f in cs_fields)
        for edu in education
    )
    features["cs_degree_flag"] = float(has_cs_degree)

    return features


# ─────────────────────────────────────────────────────────────────
# GROUP C: BEHAVIORAL & AVAILABILITY FEATURES
# ─────────────────────────────────────────────────────────────────

def compute_behavioral_features(candidate: Dict[str, Any]) -> Dict[str, float]:
    """Compute behavioral and availability features from Redrob signals."""
    features = {}
    signals = candidate.get("redrob_signals", {})
    profile = candidate.get("profile", {})

    # Activity recency — exponential decay
    last_active = signals.get("last_active_date")
    days_inactive = _days_ago(last_active)
    if days_inactive is None:
        features["recency_score"] = 0.3
    else:
        decay = math.exp(-days_inactive / INACTIVITY_HALF_LIFE_DAYS)
        features["recency_score"] = decay

    # Open to work flag
    otw = signals.get("open_to_work_flag", False)
    features["open_to_work_score"] = 1.0 if otw else 0.55

    # Recruiter response rate
    rrr = signals.get("recruiter_response_rate", 0.0)
    features["recruiter_response_rate"] = float(rrr)

    # Response time (lower is better)
    avg_response_time = signals.get("avg_response_time_hours", 72)
    if avg_response_time <= 8:
        features["response_time_score"] = 1.0
    elif avg_response_time <= 24:
        features["response_time_score"] = 0.90
    elif avg_response_time <= 48:
        features["response_time_score"] = 0.75
    elif avg_response_time <= 96:
        features["response_time_score"] = 0.55
    else:
        features["response_time_score"] = 0.3

    # Notice period
    notice = signals.get("notice_period_days", 60)
    if notice <= 0:
        features["notice_period_score"] = 1.0  # immediate joiner
    elif notice <= IDEAL_NOTICE_PERIOD_DAYS:
        features["notice_period_score"] = 1.0
    elif notice <= 60:
        # Interpolate between 1.0 (at 30) and 0.75 (at 60)
        features["notice_period_score"] = 1.0 - ((notice - IDEAL_NOTICE_PERIOD_DAYS) / (60.0 - IDEAL_NOTICE_PERIOD_DAYS)) * 0.25
    elif notice <= MAX_NOTICE_PERIOD_DAYS:
        # Interpolate between 0.75 (at 60) and 0.5 (at 90)
        features["notice_period_score"] = 0.75 - ((notice - 60) / (MAX_NOTICE_PERIOD_DAYS - 60.0)) * 0.25
    else:
        # Interpolate/decay between 0.5 (at 90) and 0.2 (at 180)
        features["notice_period_score"] = max(0.2, 0.5 - ((notice - MAX_NOTICE_PERIOD_DAYS) / 90.0) * 0.3)

    # Relocation willingness
    features["relocation_score"] = 1.0 if signals.get("willing_to_relocate", False) else 0.65

    # Location match
    location = profile.get("location", "").lower()
    country = profile.get("country", "").lower()
    if any(loc in location for loc in PREFERRED_LOCATIONS):
        features["location_score"] = 1.0
    elif "india" in country or any(loc in location for loc in INDIA_LOCATIONS):
        features["location_score"] = 0.8
    else:
        features["location_score"] = 0.5

    # Profile views (platform engagement signal — popular profiles get seen more)
    views = signals.get("profile_views_received_30d", 0)
    features["profile_views_score"] = min(1.0, views / 100.0)

    # Recruiter saves (strongest signal — recruiters actively saved this profile)
    saves = signals.get("saved_by_recruiters_30d", 0)
    features["recruiter_saves_score"] = min(1.0, saves / 10.0)

    # Search appearances
    appearances = signals.get("search_appearance_30d", 0)
    features["search_appearances_score"] = min(1.0, appearances / 500.0)

    # Interview completion rate (reliability signal)
    icr = signals.get("interview_completion_rate", 0.5)
    features["interview_completion_rate"] = float(icr)

    # Offer acceptance rate
    oar = signals.get("offer_acceptance_rate", -1)
    if oar == -1:
        features["offer_acceptance_score"] = 0.0  # missing -> weight=0 (near-zero)
    elif oar == 0.0:
        features["offer_acceptance_score"] = 0.1  # 0.0 -> strong negative
    else:
        features["offer_acceptance_score"] = float(oar)

    # Profile completeness
    completeness = signals.get("profile_completeness_score", 0)
    features["profile_completeness"] = completeness / 100.0

    # Verification score
    ver_email = int(signals.get("verified_email", False))
    ver_phone = int(signals.get("verified_phone", False))
    ver_linkedin = int(signals.get("linkedin_connected", False))
    features["verification_score"] = (ver_email + ver_phone + ver_linkedin) / 3.0

    # Connection count (professional network)
    connections = signals.get("connection_count", 0)
    features["connections_score"] = min(1.0, connections / 500.0)

    # Applications submitted (shows active job seeking)
    apps = signals.get("applications_submitted_30d", 0)
    features["applications_score"] = min(1.0, apps / 10.0)

    return features


# ─────────────────────────────────────────────────────────────────
# GROUP D: RISK & CONSISTENCY FEATURES
# ─────────────────────────────────────────────────────────────────

def compute_risk_features(candidate: Dict[str, Any]) -> Dict[str, float]:
    """Compute risk and profile quality features."""
    features = {}
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})
    profile = candidate.get("profile", {})

    # Career gap penalty: sum of months between consecutive roles
    gap_months = _compute_career_gaps(career)
    features["career_gap_months"] = gap_months
    features["career_gap_score"] = max(0.0, 1.0 - gap_months / 36.0)  # 36+ months = 0.0

    # Skill endorsement ratio
    if skills:
        endorsements = signals.get("endorsements_received", 0)
        features["endorsement_per_skill"] = min(1.0, endorsements / (len(skills) * 5))
    else:
        features["endorsement_per_skill"] = 0.0

    # Consulting-only flag (already computed in career features, but kept here for transparency)
    all_service = all(
        _classify_company(r.get("company", "")) == "service"
        for r in career
    )
    features["is_consulting_only"] = float(all_service)

    # Profile age (older = more likely legitimate)
    signup_date = signals.get("signup_date")
    signup_days = _days_ago(signup_date)
    if signup_days is None:
        features["profile_age_score"] = 0.5
    else:
        features["profile_age_score"] = min(1.0, signup_days / 365.0)  # 1+ year = 1.0

    return features


def _compute_career_gaps(career: List[Dict]) -> float:
    """Compute total gap months between consecutive roles."""
    if len(career) < 2:
        return 0.0

    # Sort by start date
    dated_career = []
    for role in career:
        sd = _parse_date(role.get("start_date"))
        ed = _parse_date(role.get("end_date")) if not role.get("is_current") else EVAL_DATE
        if sd and ed:
            dated_career.append((sd, ed))

    if len(dated_career) < 2:
        return 0.0

    dated_career.sort(key=lambda x: x[0])

    total_gap_months = 0.0
    for i in range(1, len(dated_career)):
        prev_end = dated_career[i - 1][1]
        curr_start = dated_career[i][0]
        gap_days = (curr_start - prev_end).days
        if gap_days > 30:  # ignore minor overlaps/short breaks
            total_gap_months += gap_days / 30.0

    return total_gap_months


# ─────────────────────────────────────────────────────────────────
# GROUP E: LATENT / DERIVED FEATURES
# ─────────────────────────────────────────────────────────────────

def compute_latent_features(
    career_feats: Dict[str, float],
    tech_feats: Dict[str, float],
    behavioral_feats: Dict[str, float],
    risk_feats: Dict[str, float],
) -> Dict[str, float]:
    """Compute derived latent features from other feature groups."""
    features = {}

    # is_shipper: concrete evidence of building + deploying at product companies
    is_shipper = (
        career_feats.get("shipper_verb_score", 0) > 0.2
        and career_feats.get("product_company_years", 0) > 1.5
        and tech_feats.get("production_ml_score", 0) > 0.1
    )
    features["is_shipper"] = float(is_shipper)

    # technical_depth_score: multi-signal measure of actual technical depth
    features["technical_depth_score"] = (
        0.35 * tech_feats.get("core_skill_weighted_score", 0)
        + 0.25 * tech_feats.get("description_technical_depth", 0)
        + 0.20 * tech_feats.get("github_score", 0)
        + 0.20 * tech_feats.get("avg_assessment_score", 0)
    )

    # recruiter_fit_score: how "recruitable" this person actually is
    features["recruiter_fit_score"] = (
        0.35 * behavioral_feats.get("recruiter_response_rate", 0)
        + 0.25 * behavioral_feats.get("notice_period_score", 0)
        + 0.25 * behavioral_feats.get("open_to_work_score", 0)
        + 0.15 * behavioral_feats.get("interview_completion_rate", 0)
    )

    # availability_composite: composite availability signal
    features["availability_composite"] = (
        0.30 * behavioral_feats.get("notice_period_score", 0)
        + 0.25 * behavioral_feats.get("open_to_work_score", 0)
        + 0.20 * behavioral_feats.get("recency_score", 0)
        + 0.15 * behavioral_feats.get("relocation_score", 0)
        + 0.10 * behavioral_feats.get("location_score", 0)
    )

    # researcher_vs_builder: 1.0 = builder, 0.0 = researcher
    features["builder_score"] = (
        0.4 * career_feats.get("shipper_verb_score", 0)
        + 0.3 * career_feats.get("product_ratio", 0)
        + 0.3 * tech_feats.get("production_ml_score", 0)
    )

    # domain_specialist: strong in retrieval AND ranking AND LLM
    features["domain_specialist_score"] = (
        0.35 * tech_feats.get("retrieval_score", 0)
        + 0.35 * tech_feats.get("ranking_score", 0)
        + 0.30 * tech_feats.get("llm_score", 0)
    )

    # engagement_composite: platform engagement combining multiple signals
    features["engagement_composite"] = (
        0.35 * behavioral_feats.get("recruiter_saves_score", 0)
        + 0.30 * behavioral_feats.get("profile_views_score", 0)
        + 0.20 * behavioral_feats.get("search_appearances_score", 0)
        + 0.15 * behavioral_feats.get("applications_score", 0)
    )

    # behavioral_gate composite
    recency_score = behavioral_feats.get("recency_score", 0.3)
    rrr = behavioral_feats.get("recruiter_response_rate", 0.0)
    otw_score = behavioral_feats.get("open_to_work_score", 0.55)
    
    behavioral_gate = 0.40 * recency_score + 0.35 * rrr + 0.25 * otw_score
    features["behavioral_gate"] = max(0.2, min(1.0, behavioral_gate))

    return features


# ─────────────────────────────────────────────────────────────────
# GROUP F: JD–CANDIDATE INTERACTION FEATURES
# ─────────────────────────────────────────────────────────────────

# JD requirements for interaction feature computation
JD_REQUIRED_SKILLS = {
    "embedding", "embeddings", "vector search", "vector database", "faiss", "milvus",
    "weaviate", "pinecone", "qdrant", "learning to rank", "ltr", "lambdamart",
    "bm25", "information retrieval", "ranking", "elasticsearch", "opensearch",
    "llm", "large language model", "rag", "retrieval augmented generation",
    "fine-tuning", "finetuning", "fine tuning", "lora", "transformer", "bert",
    "pytorch", "nlp", "natural language processing", "mlops", "model serving",
    "model deployment", "inference", "deep learning", "machine learning",
    "recommendation", "reranking", "hugging face", "huggingface",
}

JD_REQUIRED_TITLES = {
    "machine learning engineer", "ml engineer", "ai engineer", "nlp engineer",
    "search engineer", "ranking engineer", "recommendation engineer",
    "retrieval engineer", "applied scientist", "applied ml", "research engineer",
    "data scientist", "senior data scientist", "mlops engineer",
    "staff engineer", "principal engineer", "tech lead",
}

JD_PREFERRED_LOCATION_ZONE = {"pune", "noida", "delhi", "ncr", "gurugram", "gurgaon", "greater noida"}
JD_PREFERRED_NOTICE_DAYS = 30
JD_IDEAL_YOE_RANGE = (4.0, 12.0)
JD_PREFERRED_COMPANY_SIZE = {"1-10", "11-50", "51-200", "201-500", "501-1000"}


def compute_interaction_features(candidate: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute JD–candidate interaction features.
    These measure how well the candidate matches the SPECIFIC JD requirements.
    """
    features = {}
    profile = candidate.get("profile", {})
    skills = candidate.get("skills", [])
    career = candidate.get("career_history", [])
    signals = candidate.get("redrob_signals", {})

    # ── Skill Overlap (Jaccard-like) ──────────────────────────────
    candidate_skills = {s.get("name", "").lower().strip() for s in skills}
    # Fuzzy match: check if any JD skill substring appears in candidate skills
    jd_hits = set()
    for jd_skill in JD_REQUIRED_SKILLS:
        for cs in candidate_skills:
            if jd_skill in cs or cs in jd_skill:
                jd_hits.add(jd_skill)
                break
    if len(JD_REQUIRED_SKILLS) > 0:
        features["skill_overlap_jaccard"] = len(jd_hits) / len(JD_REQUIRED_SKILLS)
        features["skill_overlap_count"] = float(len(jd_hits))
        features["skill_overlap_ratio"] = len(jd_hits) / max(1, len(candidate_skills))
    else:
        features["skill_overlap_jaccard"] = 0.0
        features["skill_overlap_count"] = 0.0
        features["skill_overlap_ratio"] = 0.0

    # ── Title Similarity ──────────────────────────────────────────
    current_title = profile.get("current_title", "").lower()
    # Check how many JD-aligned title patterns match
    title_matches = sum(1 for jt in JD_REQUIRED_TITLES if jt in current_title)
    features["title_jd_match_score"] = min(1.0, title_matches)

    # Check all career titles for JD alignment
    all_titles = [r.get("title", "").lower() for r in career]
    career_title_matches = sum(
        1 for title in all_titles
        for jt in JD_REQUIRED_TITLES
        if jt in title
    )
    features["career_title_jd_match_count"] = float(min(career_title_matches, 5))
    features["career_title_jd_match_ratio"] = career_title_matches / max(1, len(all_titles))

    # ── Notice Period Compatibility ───────────────────────────────
    notice = signals.get("notice_period_days", 60)
    if notice <= JD_PREFERRED_NOTICE_DAYS:
        features["notice_compatibility"] = 1.0
    elif notice <= 60:
        features["notice_compatibility"] = 0.7
    elif notice <= 90:
        features["notice_compatibility"] = 0.4
    else:
        features["notice_compatibility"] = 0.1

    # ── Location Compatibility ────────────────────────────────────
    location = profile.get("location", "").lower()
    country = profile.get("country", "").lower()
    relocate = signals.get("willing_to_relocate", False)
    if any(loc in location for loc in JD_PREFERRED_LOCATION_ZONE):
        features["location_compatibility"] = 1.0
    elif "india" in country and relocate:
        features["location_compatibility"] = 0.8
    elif "india" in country:
        features["location_compatibility"] = 0.6
    elif relocate:
        features["location_compatibility"] = 0.4
    else:
        features["location_compatibility"] = 0.2

    # ── YOE Compatibility ─────────────────────────────────────────
    yoe = profile.get("years_of_experience", 0)
    if JD_IDEAL_YOE_RANGE[0] <= yoe <= JD_IDEAL_YOE_RANGE[1]:
        features["yoe_compatibility"] = 1.0
    elif yoe < JD_IDEAL_YOE_RANGE[0]:
        features["yoe_compatibility"] = max(0.1, yoe / JD_IDEAL_YOE_RANGE[0])
    else:
        features["yoe_compatibility"] = max(0.5, 1.0 - (yoe - JD_IDEAL_YOE_RANGE[1]) * 0.05)

    # ── Work Mode Compatibility ───────────────────────────────────
    work_mode = signals.get("preferred_work_mode", "flexible")
    work_mode_scores = {"flexible": 1.0, "hybrid": 0.9, "remote": 0.7, "onsite": 0.6}
    features["work_mode_compatibility"] = work_mode_scores.get(work_mode, 0.5)

    # ── Salary Range Compatibility ────────────────────────────────
    salary = signals.get("expected_salary_range_inr_lpa", {})
    salary_min = salary.get("min", 0)
    salary_max = salary.get("max", 0)
    # Senior AI Engineer in Pune/Noida: ~25-60 LPA typical
    if 15 <= salary_min <= 80 and salary_max <= 120:
        features["salary_compatibility"] = 1.0
    elif salary_max > 120:
        features["salary_compatibility"] = 0.4  # overpriced
    elif salary_min < 10:
        features["salary_compatibility"] = 0.5  # too low suggests junior
    else:
        features["salary_compatibility"] = 0.7

    # ── Company Size Compatibility ────────────────────────────────
    company_size = profile.get("current_company_size", "unknown")
    if company_size in JD_PREFERRED_COMPANY_SIZE:
        features["company_size_compatibility"] = 1.0
    elif company_size in {"1001-5000", "5001-10000"}:
        features["company_size_compatibility"] = 0.7
    elif company_size == "10001+":
        features["company_size_compatibility"] = 0.5
    else:
        features["company_size_compatibility"] = 0.5

    return features


# ─────────────────────────────────────────────────────────────────
# GROUP G: ADDITIONAL SCHEMA FEATURES (Previously Unused)
# ─────────────────────────────────────────────────────────────────

def compute_additional_features(candidate: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute features from schema fields that were previously unused.
    """
    features = {}
    skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])

    # ── Weighted Skill Duration Score ─────────────────────────────
    # Use skill.duration_months weighted by JD-relevance
    total_weighted_duration = 0.0
    total_jd_skill_duration = 0.0
    for skill in skills:
        sname = skill.get("name", "").lower()
        dur = skill.get("duration_months", 0)
        for jd_skill in JD_REQUIRED_SKILLS:
            if jd_skill in sname or sname in jd_skill:
                total_jd_skill_duration += dur
                break
        total_weighted_duration += dur

    features["jd_skill_duration_months"] = total_jd_skill_duration
    features["jd_skill_duration_score"] = min(1.0, total_jd_skill_duration / 60.0)  # 60+ months = 1.0
    features["avg_skill_duration"] = total_weighted_duration / max(1, len(skills))

    # ── Industry Alignment ────────────────────────────────────────
    current_industry = profile.get("current_industry", "").lower()
    tech_industries = {"technology", "internet", "computer software", "saas", "artificial intelligence",
                       "machine learning", "information technology", "fintech", "edtech", "e-commerce"}
    service_industries = {"it services", "consulting", "staffing", "outsourcing", "bpo"}
    
    if any(ti in current_industry for ti in tech_industries):
        features["industry_alignment"] = 1.0
    elif any(si in current_industry for si in service_industries):
        features["industry_alignment"] = 0.3
    else:
        features["industry_alignment"] = 0.5

    # ── Career Industry Consistency ───────────────────────────────
    tech_roles = sum(
        1 for r in career
        if any(ti in r.get("industry", "").lower() for ti in tech_industries)
    )
    features["tech_industry_ratio"] = tech_roles / max(1, len(career))

    # ── Certifications ────────────────────────────────────────────
    certs = candidate.get("certifications", [])
    ml_cert_keywords = {"machine learning", "deep learning", "ai", "data science", "mlops",
                        "tensorflow", "pytorch", "aws", "gcp", "azure", "cloud"}
    ml_certs = sum(
        1 for c in certs
        if any(kw in c.get("name", "").lower() for kw in ml_cert_keywords)
    )
    features["ml_certification_count"] = float(ml_certs)
    features["has_ml_certification"] = float(ml_certs > 0)
    features["total_certifications"] = float(len(certs))

    # ── Skill Count Features ──────────────────────────────────────
    features["total_skill_count"] = float(len(skills))
    expert_skills = sum(1 for s in skills if s.get("proficiency") == "expert")
    advanced_skills = sum(1 for s in skills if s.get("proficiency") == "advanced")
    features["expert_skill_count"] = float(expert_skills)
    features["advanced_skill_count"] = float(advanced_skills)
    features["expert_ratio"] = expert_skills / max(1, len(skills))

    return features


# ─────────────────────────────────────────────────────────────────
# MAIN FEATURE COMPUTATION ENTRY POINT
# ─────────────────────────────────────────────────────────────────

def compute_all_features(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute the full feature vector for a candidate.
    Returns a flat dict of all features.
    """
    career_feats = compute_career_features(candidate)
    tech_feats = compute_technical_features(candidate)
    behavioral_feats = compute_behavioral_features(candidate)
    risk_feats = compute_risk_features(candidate)
    latent_feats = compute_latent_features(
        career_feats, tech_feats, behavioral_feats, risk_feats
    )
    interaction_feats = compute_interaction_features(candidate)
    additional_feats = compute_additional_features(candidate)

    all_features = {}
    all_features.update({f"career_{k}": v for k, v in career_feats.items()})
    all_features.update({f"tech_{k}": v for k, v in tech_feats.items()})
    all_features.update({f"behavioral_{k}": v for k, v in behavioral_feats.items()})
    all_features.update({f"risk_{k}": v for k, v in risk_feats.items()})
    all_features.update({f"latent_{k}": v for k, v in latent_feats.items()})
    all_features.update({f"interaction_{k}": v for k, v in interaction_feats.items()})
    all_features.update({f"additional_{k}": v for k, v in additional_feats.items()})

    return all_features

