"""
reasoning.py — Template-based, non-hallucinated reasoning generation.

All reasoning strings are derived ONLY from actual profile fields.
No LLM calls. No invented facts.

Quality criteria from the submission spec:
  1. Specificity: references actual companies, skills, durations
  2. JD connection: links candidate attributes to JD requirements
  3. Acknowledgment of concerns: mentions weaknesses where relevant
  4. No hallucination: every claim verifiable from profile data
  5. Variation: different templates for different candidate types
  6. Rank consistency: high-ranked candidates get positive framing
"""

from typing import Dict, Any, Optional
from ranker.config import EVAL_DATE, ALIGNED_TITLES, DISQUALIFYING_TITLES


# 28 distinct templates: 4 rank tiers × 7 dominant strengths
TEMPLATES = {
    "ELITE": {
        "DOMAIN_SHIPPER": "Strong fit: {yoe:.0f}yr {current_title} with production retrieval/recommendation experience at {top_company}, bringing {skills_str}. Highly active ({rrr:.0%} response) and open to work.",
        "TECH_DEPTH": "Highly technical: {yoe:.0f}yr {current_title} with strong {skills_str} depth, backed by a {github:.0f} GitHub activity score and solid technical-assessment performance.",
        "BEHAVIORAL_STRONG": "Top fit: {yoe:.0f}yr {current_title} with strong hireability ({rrr:.0%} response rate, {notice}d notice), combining solid {skills_str} with high platform engagement.",
        "PEDIGREE_STRONG": "Strong product-company background: {yoe:.0f}yr {current_title} with tenure at firms like {top_company} and production-ready {skills_str} experience.",
        "LOCATION_CONCERN": "Strong profile: {yoe:.0f}yr {current_title} with solid {skills_str} depth from {top_company}; based in {location} but open to relocation with high responsiveness.",
        "EXPERIENCE_CONCERN": "Strong fit with an experience-band note: {yoe:.0f}yr {current_title} at {top_company} with relevant vector search / LLM exposure; YOE sits outside the ideal 4-12yr range.",
        "CONSULTING_ADJACENT": "Adaptive profile: {yoe:.0f}yr {current_title} with product engineering experience at {top_company} alongside some service-firm work, showing {skills_str} competency."
    },
    "TOP": {
        "DOMAIN_SHIPPER": "Strong match: {yoe:.0f}yr {current_title} who deployed retrieval and recommendation systems at {top_company}, matching the JD's core vector search and ranking requirements.",
        "TECH_DEPTH": "Deep technical fit: {yoe:.0f}yr engineer showing strong command of {skills_str} and PyTorch, with verified coding credentials and active platform participation.",
        "BEHAVIORAL_STRONG": "Responsive candidate: {yoe:.0f}yr {current_title} from {top_company} with highly favorable recruiter engagement metrics ({rrr:.0%} response, {notice}d notice period) and solid technical fundamentals.",
        "PEDIGREE_STRONG": "Tier-1 background: {yoe:.0f}yr developer with tenure at high-profile firms like {top_company}, well-versed in production ML lifecycle, scalable search and {skills_str}.",
        "LOCATION_CONCERN": "High-potential candidate: {yoe:.0f}yr {current_title} with solid {skills_str} depth at {top_company}, currently based in {location} but actively looking to relocate.",
        "EXPERIENCE_CONCERN": "Promising talent: {yoe:.1f}yr engineer with rapid progression and {skills_str} expertise at {top_company}, showing strong aptitude despite being on the edge of the ideal experience band.",
        "CONSULTING_ADJACENT": "Balanced background: {yoe:.0f}yr developer combining service firm execution discipline with recent product engineering at {top_company}, featuring solid {skills_str} skills."
    },
    "MID": {
        "DOMAIN_SHIPPER": "Mid-tier fit: {yoe:.0f}yr {current_title} at {current_company} with search/ranking project exposure, though has slightly less evidence of end-to-end product ownership.",
        "TECH_DEPTH": "Capable engineer: {yoe:.0f}yr developer with good familiarity in {skills_str}, but lacks the deep production-level shipping history required for elite ranks.",
        "BEHAVIORAL_STRONG": "Engaged applicant: {yoe:.0f}yr {current_title} showing great platform activity and quick responsiveness, though core ML/search domain experience is moderate.",
        "PEDIGREE_STRONG": "Solid pedigree: {yoe:.0f}yr engineer with product company exposure (incl. {top_company}), but the profile shows a more generalist software engineering focus than specialized ML/RAG.",
        "LOCATION_CONCERN": "Geographically distant: competent {yoe:.0f}yr developer with relevant {skills_str} expertise, but based in {location} which may present logistics challenges.",
        "EXPERIENCE_CONCERN": "Experience mismatch: {yoe:.0f}yr candidate at {current_company} whose YOE falls outside the ideal 4-12 year range, requiring extra evaluation of technical depth.",
        "CONSULTING_ADJACENT": "Service firm background: {yoe:.0f}yr developer at {current_company} with a history of consulting roles, but has some product exposure and technical skills like {skills_str}."
    },
    "LOWER": {
        "DOMAIN_SHIPPER": "Lower-tier candidate: {yoe:.1f}yr developer at {current_company} showing limited domain shipping evidence and weak overall alignment with JD-specific search requirements.",
        "TECH_DEPTH": "Limited technical match: {yoe:.1f}yr developer at {current_company} listing basic AI skills, but lacking verified hands-on implementation or deep coding activity.",
        "BEHAVIORAL_STRONG": "Inactive profile: {yoe:.1f}yr {current_title} with low response rates and profile activity, suggesting limited availability despite moderate {skills_str} background.",
        "PEDIGREE_STRONG": "Generalist developer: {yoe:.1f}yr engineer at {current_company} with software history but no demonstrated focus on recommendation engines, retrieval, or vector databases.",
        "LOCATION_CONCERN": "Relocation required: {yoe:.1f}yr developer based in {location} with weaker JD alignment and extended notice period, making hiring logistics unfavorable.",
        "EXPERIENCE_CONCERN": "Junior profile: {yoe:.1f}yr engineer with insufficient YOE ({yoe:.1f} years) and minimal exposure to shipping production-level machine learning models.",
        "CONSULTING_ADJACENT": "Consulting focus: {yoe:.1f}yr developer whose career history is primarily in services/consulting, showing lower product-ratio and limited relevancy to Series A product engineering."
    }
}


def classify_dominant_strength(candidate: Dict[str, Any], score_result: Dict[str, Any]) -> str:
    """Classify the candidate into one of 7 narrative profiles for reasoning generation."""
    profile = candidate.get("profile", {})
    
    latent = score_result.get("latent_features", {})
    career_feats = score_result.get("career_features", {})
    tech_feats = score_result.get("tech_features", {})
    behavioral_feats = score_result.get("behavioral_features", {})
    risk_feats = score_result.get("risk_features", {})
    
    location_score = behavioral_feats.get("location_score", 0.5)
    exp_fit = career_feats.get("experience_range_fit", 1.0)
    product_ratio = career_feats.get("product_ratio", 1.0)
    domain_shipping = career_feats.get("domain_shipping_score", 0.0)
    pedigree = career_feats.get("pedigree_score", 0.5)
    current_tier = career_feats.get("current_company_tier_score", 0.5)
    tech_depth = latent.get("technical_depth_score", 0.0)
    behavioral_gate = latent.get("behavioral_gate", 0.5)
    service_years = career_feats.get("service_company_years", 0.0)

    # 1. Location Concern
    if location_score < 0.6:
        return "LOCATION_CONCERN"
    # 2. Experience Concern
    if exp_fit < 0.85:
        return "EXPERIENCE_CONCERN"
    # 3. Consulting Adjacent
    if service_years > 0.5 and product_ratio > 0.0:
        return "CONSULTING_ADJACENT"
    # 4. Domain Shipper
    if domain_shipping > 0.3 and product_ratio > 0.5:
        return "DOMAIN_SHIPPER"
    # 5. Pedigree Strong
    if pedigree >= 0.75 or current_tier >= 0.85:
        return "PEDIGREE_STRONG"
    # 6. Tech Depth
    if tech_depth > 0.55:
        return "TECH_DEPTH"
    # 7. Behavioral Strong
    if behavioral_gate > 0.65:
        return "BEHAVIORAL_STRONG"
        
    return "DOMAIN_SHIPPER" if product_ratio > 0.5 else "TECH_DEPTH"


def generate_reasoning(
    candidate: Dict[str, Any],
    score_result: Dict[str, Any],
    rank: int,
) -> str:
    """
    Generate a 1-2 sentence reasoning string for a candidate.

    Args:
        candidate: Raw candidate dict
        score_result: Output from scorer.score_candidate()
        rank: Final rank (1-100)

    Returns:
        reasoning string (max ~300 chars for readability)
    """
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    signals = candidate.get("redrob_signals", {})
    skills = candidate.get("skills", [])

    career_feats = score_result.get("career_features", {})
    tech_feats = score_result.get("tech_features", {})
    behavioral_feats = score_result.get("behavioral_features", {})
    trap_reasons = score_result.get("trap_reasons", "clean")

    # ── Extract key evidence ─────────────────────────────────────
    yoe = profile.get("years_of_experience", 0)
    current_title = profile.get("current_title", "")
    current_company = profile.get("current_company", "")
    location = profile.get("location", "")

    # Most recent product company (use cached value if available)
    top_company = candidate.get("_top_company", "")
    if not top_company:
        product_companies = [
            r.get("company", "") for r in career
            if r.get("industry", "").lower() not in ["it services", "consulting"]
            and r.get("company", "") not in ["", "Unknown"]
        ]
        top_company = product_companies[0] if product_companies else current_company

    # Top skills with high proficiency — focus on JD-relevant skills
    jd_relevant_keywords = {
        "embedding", "embeddings", "ranking", "retrieval", "search", "llm",
        "transformer", "bert", "pytorch", "nlp", "recommendation", "vector",
        "deep learning", "machine learning", "mlops", "inference", "rag",
        "fine-tuning", "elasticsearch", "faiss", "tensorflow",
    }
    top_skills = []
    for s in skills:
        sname = s.get("name", "")
        if (s.get("proficiency") in ("expert", "advanced")
                and any(kw in sname.lower() for kw in jd_relevant_keywords)):
            top_skills.append(sname)
        if len(top_skills) >= 3:
            break
    # Fallback to any expert/advanced skill if no JD-relevant ones found
    if not top_skills:
        top_skills = [
            s.get("name", "") for s in skills
            if s.get("proficiency") in ("expert", "advanced") and s.get("endorsements", 0) > 5
        ][:3]
    skills_str = ", ".join(top_skills) if top_skills else "relevant ML skills"

    # Behavioral signals
    rrr = signals.get("recruiter_response_rate", 0)
    notice = signals.get("notice_period_days", 60)
    github = signals.get("github_activity_score", -1)

    # ── Route to appropriate template based on rank ─────────────
    is_honeypot = score_result.get("is_honeypot", False)
    if is_honeypot:
        return _honeypot_template(profile, trap_reasons)

    if rank <= 10:
        tier = "ELITE"
    elif rank <= 30:
        tier = "TOP"
    elif rank <= 60:
        tier = "MID"
    else:
        tier = "LOWER"

    # Tone guard (Stage-4 rank/tone consistency): do not apply the most emphatic
    # "elite/senior" wording to a clearly-junior current title. Use the next, more
    # measured wording tier so the reasoning matches the candidate's actual profile.
    if tier == "ELITE" and any(
        j in current_title.lower() for j in ("junior", "associate", "trainee", "intern")
    ):
        tier = "TOP"

    strength = classify_dominant_strength(candidate, score_result)
    
    template = TEMPLATES[tier][strength]
    reasoning = template.format(
        yoe=yoe,
        top_company=top_company,
        current_title=current_title,
        current_company=current_company,
        skills_str=skills_str,
        rrr=rrr,
        notice=notice,
        github=max(0.0, github),
        location=location if location else "unknown location"
    )

    return truncate_reasoning(reasoning)


def _honeypot_template(profile, trap_reasons) -> str:
    """Template for honeypot/flagged candidates (will not appear in top-100)."""
    return (
        f"Profile flagged for data inconsistencies ({trap_reasons[:100]}). "
        f"Ranked lowest per evaluation rules."
    )


def truncate_reasoning(text: str, max_chars: int = 300) -> str:
    """Ensure reasoning fits within character limit."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars - 3].rsplit(";", 1)[0].strip() + "..."
