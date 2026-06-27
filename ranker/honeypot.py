"""
honeypot.py — Trap detection and profile consistency checking.

The dataset contains ~80 honeypot candidates with impossible/inconsistent profiles.
Submissions with >10 honeypots in the top-100 are DISQUALIFIED.

Detection logic is divided into:
1. Temporal impossibility checks (experience > possible career length)
2. Skill bloat checks (expert in too many skills with zero usage history)
3. Behavioral implausibility checks (impossible signal combinations)
4. Keyword stuffer detection (AI buzzwords without matching career history)
"""

import re
from datetime import date, datetime
from typing import Dict, Any, Tuple

from ranker.config import EVAL_DATE, DISQUALIFYING_TITLES, CORE_SKILLS

# Real-world founding years of all companies in the dataset
COMPANY_FOUNDING_YEARS = {
    "Krutrim": 2023,
    "Sarvam AI": 2023,
    "Rephrase.ai": 2019,
    "Saarthi.ai": 2017,
    "Observe.AI": 2017,
    "Yellow.ai": 2016,
    "Wysa": 2015,
    "Verloop.io": 2015,
    "Niramai": 2016,
    "Glance": 2019,
    "Locobuzz": 2015,
    "Aganitha": 2017,
    "Mad Street Den": 2013,
    "Zepto": 2021,
    "CRED": 2018,
    "Redrob": 2023,
    "PhonePe": 2015,
    "Meesho": 2015,
    "Nykaa": 2012,
    "BYJU'S": 2011,
    "Vedantu": 2011,
    "Unacademy": 2015,
    "PharmEasy": 2015,
    "upGrad": 2015,
    "Dream11": 2008,
    "Freshworks": 2010,
    "Ola": 2010,
    "Paytm": 2010,
    "Swiggy": 2014,
    "Razorpay": 2014,
    "Zomato": 2008,
    "Flipkart": 2007,
    "InMobi": 2007,
    "PolicyBazaar": 2008,
    "Mindtree": 1999,
    "Mphasis": 1992,
    "Cognizant": 1994,
    "Capgemini": 1967,
    "Accenture": 1989,
    "HCL": 1976,
    "Tech Mahindra": 1986,
    "Wipro": 1945,
    "Infosys": 1981,
    "TCS": 1968,
    "Google": 1998,
    "Netflix": 1997,
    "Amazon": 1994,
    "Salesforce": 1999,
    "Uber": 2009,
    "Meta": 2004,
    "Adobe": 1982,
    "Microsoft": 1975,
    "Apple": 1976,
    "LinkedIn": 2002,
}


def detect_honeypot(candidate: Dict[str, Any]) -> Tuple[float, str]:
    """
    Run all honeypot and consistency checks on a candidate.

    Returns:
        (multiplier, reason)
        multiplier: 1.0 = clean, 0.0 = definite honeypot, 0.1–0.5 = suspicious
        reason: human-readable explanation of any penalty applied
    """
    reasons = []
    multiplier = 1.0

    profile = candidate.get("profile", {})
    career_history = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})

    yoe = profile.get("years_of_experience", 0)

    # ─────────────────────────────────────────
    # CHECK 1: Temporal impossibility
    # If claimed YOE > plausible career length since earliest edu end
    # ─────────────────────────────────────────
    education = candidate.get("education", [])
    earliest_grad_year = None
    for edu in education:
        ey = edu.get("end_year")
        if ey and isinstance(ey, int):
            if earliest_grad_year is None or ey < earliest_grad_year:
                earliest_grad_year = ey

    if earliest_grad_year:
        max_possible_yoe = (EVAL_DATE.year - earliest_grad_year) + 2  # +2 buffer
        if yoe > max_possible_yoe + 3:  # hard threshold: >3 years over max
            multiplier = 0.0
            reasons.append(
                f"HONEYPOT: claimed {yoe:.1f}yr YOE but graduated {earliest_grad_year} "
                f"(max possible ~{max_possible_yoe}yrs)"
            )
            return multiplier, "; ".join(reasons)

    # ─────────────────────────────────────────
    # CHECK 2: Career history start date impossibility
    # First job start date must be after roughly age 18-20
    # ─────────────────────────────────────────
    career_start_dates = []
    for role in career_history:
        sd = role.get("start_date")
        if sd:
            try:
                career_start_dates.append(datetime.strptime(sd[:10], "%Y-%m-%d").date())
            except ValueError:
                pass

    if career_start_dates:
        earliest_career = min(career_start_dates)
        if earliest_grad_year and earliest_career.year < earliest_grad_year - 2:
            # Started career more than 2 years before graduating — suspicious
            multiplier = min(multiplier, 0.85)
            reasons.append(
                f"Suspicious: career started {earliest_career.year} "
                f"but graduated {earliest_grad_year}"
            )

    # ─────────────────────────────────────────
    # CHECK 3: Skill duration vs career length
    # Total skill duration_months should not massively exceed career months
    # ─────────────────────────────────────────
    if yoe > 0:
        total_skill_months = sum(s.get("duration_months", 0) for s in skills)
        career_months = yoe * 12
        # Allow up to 3x (skills can be used concurrently across roles)
        if total_skill_months > career_months * 5 and len(skills) > 5:
            multiplier = min(multiplier, 0.80)
            reasons.append(
                f"Suspicious skill duration: {total_skill_months}mo total skill usage "
                f"vs {career_months:.0f}mo career"
            )

    # ─────────────────────────────────────────
    # CHECK 4: Expert skill bloat with zero endorsements
    # Many "expert" skills but no endorsements = fabricated
    # ─────────────────────────────────────────
    if skills:
        expert_skills = [s for s in skills if s.get("proficiency") == "expert"]
        total_endorsements = signals.get("endorsements_received", 0)
        if len(expert_skills) > 10 and total_endorsements < 5:
            multiplier = min(multiplier, 0.75)
            reasons.append(
                f"Suspicious: {len(expert_skills)} expert skills with only "
                f"{total_endorsements} total endorsements"
            )

    # ─────────────────────────────────────────
    # CHECK 5: Perfect profile completeness with zero activity
    # 100% completeness + zero views + zero connections = synthetic bot
    # ─────────────────────────────────────────
    completeness = signals.get("profile_completeness_score", 0)
    views_30d = signals.get("profile_views_received_30d", 0)
    connections = signals.get("connection_count", 0)
    if completeness >= 99.0 and views_30d == 0 and connections < 10:
        multiplier = min(multiplier, 0.80)
        reasons.append(
            f"Suspicious: 100% complete profile with {views_30d} views and "
            f"{connections} connections"
        )

    # ─────────────────────────────────────────
    # CHECK 6: Career history duration mismatch
    # Sum of duration_months in career history should ~= YOE * 12 ± 24 months
    # ─────────────────────────────────────────
    if career_history:
        total_career_months = sum(r.get("duration_months", 0) for r in career_history)
        expected_months = yoe * 12
        discrepancy = abs(total_career_months - expected_months)
        if discrepancy > 60 and yoe > 3:  # >5 year discrepancy
            multiplier = min(multiplier, 0.85)
            reasons.append(
                f"Career duration mismatch: {total_career_months}mo in history "
                f"vs {expected_months:.0f}mo from YOE={yoe}"
            )

    # ─────────────────────────────────────────
    # CHECK 7: Future dates in career history
    # ─────────────────────────────────────────
    for role in career_history:
        sd = role.get("start_date", "")
        if sd:
            try:
                start = datetime.strptime(sd[:10], "%Y-%m-%d").date()
                if start > EVAL_DATE:
                    multiplier = 0.0
                    reasons.append(f"HONEYPOT: future start date {sd}")
                    return multiplier, "; ".join(reasons)
            except ValueError:
                pass

    # ─────────────────────────────────────────
    # CHECK 8: Company founding year violation
    # ─────────────────────────────────────────
    for role in career_history:
        comp = role.get("company")
        if comp in COMPANY_FOUNDING_YEARS:
            f_year = COMPANY_FOUNDING_YEARS[comp]
            sd = role.get("start_date", "")
            if sd:
                try:
                    start_year = datetime.strptime(sd[:10], "%Y-%m-%d").year
                    if start_year < f_year:
                        multiplier = 0.0
                        reasons.append(
                            f"HONEYPOT: worked at {comp} in {start_year} "
                            f"but company was founded in {f_year}"
                        )
                        return multiplier, "; ".join(reasons)
                except ValueError:
                    pass

    reason_str = "; ".join(reasons) if reasons else "clean"
    return multiplier, reason_str


def detect_keyword_stuffer(candidate: Dict[str, Any]) -> Tuple[float, str]:
    """
    Detect keyword stuffers: candidates who list AI skills but have no
    matching experience in career descriptions.

    Returns:
        (multiplier, reason)
    """
    skills = candidate.get("skills", [])
    career_history = candidate.get("career_history", [])
    profile = candidate.get("profile", {})

    # Count AI/ML skill names in skills list
    skill_names_lower = {s.get("name", "").lower() for s in skills}
    ai_skill_hits_in_list = sum(
        1 for kw in CORE_SKILLS
        if kw in skill_names_lower or any(kw in sn for sn in skill_names_lower)
    )

    # Count AI/ML keyword mentions in actual career descriptions
    all_descriptions = " ".join(
        r.get("description", "") for r in career_history
    ).lower()
    ai_mentions_in_career = sum(
        1 for kw in CORE_SKILLS if kw in all_descriptions
    )

    # Check current title alignment
    current_title = profile.get("current_title", "").lower()
    is_disqualifying_title = any(
        dt in current_title for dt in DISQUALIFYING_TITLES
    )

    # If AI skills listed but zero mention in actual work → stuffer
    if is_disqualifying_title:
        if ai_skill_hits_in_list >= 5 and ai_mentions_in_career == 0:
            return 0.4, (
                f"Keyword stuffer: {ai_skill_hits_in_list} AI skills listed "
                f"but 0 mentions in career history (title: {profile.get('current_title')})"
            )
        if ai_skill_hits_in_list >= 8 and ai_mentions_in_career < 2:
            return 0.5, (
                f"Possible stuffer: {ai_skill_hits_in_list} AI skills, "
                f"only {ai_mentions_in_career} career mentions (title: {profile.get('current_title')})"
            )
    else:
        # For neutral titles (like Software Engineer), if they stuff 8+ AI skills but have 0 career mentions
        if ai_skill_hits_in_list >= 8 and ai_mentions_in_career == 0:
            return 0.6, (
                f"Neutral-title stuffer: {ai_skill_hits_in_list} AI skills listed "
                f"but 0 mentions in career history (title: {profile.get('current_title')})"
            )

    return 1.0, "clean"


def get_trap_multipliers(candidate: Dict[str, Any]) -> Tuple[float, float, str]:
    """
    Run all trap detection and return combined multipliers.

    Returns:
        (honeypot_mult, stuffer_mult, combined_reason)
    """
    hp_mult, hp_reason = detect_honeypot(candidate)
    st_mult, st_reason = detect_keyword_stuffer(candidate)

    all_reasons = []
    if hp_reason != "clean":
        all_reasons.append(hp_reason)
    if st_reason != "clean":
        all_reasons.append(st_reason)

    return hp_mult, st_mult, " | ".join(all_reasons) if all_reasons else "clean"
