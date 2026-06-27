import csv
import json
import os
from tqdm import tqdm  # type: ignore

def main():
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    final_dir = os.path.dirname(scripts_dir)
    csv_path = os.path.join(final_dir, "output/submission.csv")
    candidates_jsonl = os.path.join(final_dir, "input/candidates.jsonl")
    output_dir = os.path.join(final_dir, "output/top_100_resumes")

    # 1. Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"[resumes] Creating markdown resumes in: {output_dir}")

    # 2. Load top-100 candidate IDs and ranks/scores
    top_100_ranks = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            cid = r["candidate_id"]
            top_100_ranks[cid] = {
                "rank": r["rank"],
                "score": r["score"],
                "reasoning": r["reasoning"]
            }

    # 3. Stream candidates.jsonl and generate resumes for the top-100
    generated_count = 0
    with open(candidates_jsonl, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Generating Resumes", unit="cand"):
            if not line.strip():
                continue
            cand = json.loads(line)
            cid = cand["candidate_id"]
            
            if cid in top_100_ranks:
                rank_info = top_100_ranks[cid]
                
                # Extract candidate details
                profile = cand.get("profile", {})
                career = cand.get("career_history", [])
                skills = cand.get("skills", [])
                signals = cand.get("redrob_signals", {})
                
                skill_names = [s.get("name", "") for s in skills if isinstance(s, dict)]
                
                # Format career history
                career_lines = []
                for role in career:
                    title = role.get("title", "N/A")
                    company = role.get("company", "N/A")
                    duration = role.get("duration_months", 0)
                    desc = role.get("description", "")
                    
                    career_lines.append(f"### {title} | {company}")
                    career_lines.append(f"*Duration: {duration} months*\n")
                    if desc:
                        career_lines.append(f"{desc.strip()}\n")
                    career_lines.append("")

                career_history_str = "\n".join(career_lines)

                # Assemble Resume Markdown
                resume_content = f"""# Candidate Resume: {cid} (Rank {rank_info['rank']})

**Current Title:** {profile.get('current_title', 'N/A')} at {profile.get('current_company', 'N/A')}
**Total Experience:** {profile.get('years_of_experience', 'N/A')} Years
**Current Location:** {profile.get('location', 'N/A')}

---

## Ranker Assessment Details
- **Rank:** #{rank_info['rank']}
- **Scoring Fit:** {rank_info['score']}
- **Summary Feedback:** *{rank_info['reasoning']}*

---

## Contact & Availability Signals
- **Recruiter Response Rate:** {signals.get('recruiter_response_rate', 'N/A')}
- **Notice Period:** {signals.get('notice_period_days', 'N/A')} days
- **Open to Work Status:** {signals.get('open_to_work_flag', 'N/A')}

---

## Technical Skills & Expertise
{", ".join(skill_names) if skill_names else "None listed"}

---

## Professional Career History

{career_history_str if career_history_str else "No career history details listed."}

---
*Standardized Resume generated on July 27 for manual profile verification.*
"""
                
                # Write resume file
                filename = f"Rank_{int(rank_info['rank']):03d}_{cid}.md"
                filepath = os.path.join(output_dir, filename)
                with open(filepath, "w", encoding="utf-8") as out:
                    out.write(resume_content)
                generated_count += 1

    print(f"[resumes] Successfully generated {generated_count} resumes.")

if __name__ == "__main__":
    main()
