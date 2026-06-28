# Changes — Reproducibility Fixes & Ranking-Quality Improvements

This document records the changes made to the Redrob hackathon submission, grouped by
intent. The work fixes issues that could have eliminated the submission before scoring
(Stages 1/3/4) and improves the ranking against explicit Job-Description requirements.

All changes were validated by re-running the full pipeline end-to-end on the 100k
candidate pool (`precompute → train_ltr → rank → validate`).

---

## TL;DR — verified results

- `validate_submission.py` → **"Submission is valid."** (100 rows, unique ranks 1–100, non-increasing scores).
- **Reproduction verified**: with `bm25_index.pkl` deleted (fresh-clone scenario), the documented
  command loads the **LTR model**, auto-rebuilds the BM25 index, runs in ~43s, and produces an
  **identical** ranking. Previously it silently fell back to the heuristic scorer.
- Ranking now respects the JD's explicit anti-signals (measured on the real 100k):

  | Metric (top-100) | Before | After |
  |---|---|---|
  | `domain_mismatch > 0.45` (CV/speech/robotics-primary) | 12 | **1** |
  | CV/speech/robotics-titled | 5 | 2 *(both retain real NLP/IR work — the JD carve-out)* |
  | `service_ratio > 0.5` (consulting-heavy) | 3 | **0** |
  | Honeypots | 0 | 0 |

- Reasoning: **100/100 unique** strings, tone calibrated to rank (no "Junior ML Engineer = elite veteran").

---

## Tier 1 — Reproducibility / disqualification blockers

**`rank.py`**
- Decoupled LTR-model loading from the presence of `bm25_index.pkl`. The model + feature matrix
  are now the artifacts that define the "fast" path; the BM25 index is loaded if present and
  **auto-built from `--candidates` (and cached) when absent**. Previously the model was only
  loaded *inside* the `if has_precomputed:` (bm25-pickle) gate, so a repo that shipped the model
  but not the (git-ignored) pickle silently degraded to the heuristic scorer — a different ranking.
- Fixed a latent crash: the old slow path asserted `features_df is not None`, which would have
  failed on any reproduction that hit the fallback.
- Added a loud warning if `ltr_model.txt` is genuinely missing (instead of silently using heuristics).

**`requirements.txt`** — added `lightgbm` and `scipy` (the LTR model import would otherwise crash a clean install).

**`.gitignore`** — `output/submission.csv` is now committed; `input/bm25_index.pkl` (~120 MB, exceeds
GitHub's 100 MB limit) stays ignored because it is regenerable (`precompute.py`, or rank.py's auto-build).

**`submission_metadata.yaml`** — corrected `github_repo` to this repo (was `trail_Repo`),
set `pre_computation_required: true` with realistic minutes, refreshed the methodology summary,
and flagged the still-required `phone` / member email / `sandbox_link` fields.

**`README.md`** — documented the two-step `precompute.py → rank.py` flow, noted `lightgbm` is
required, and added a `docker run` recipe to satisfy the §10.5 sandbox requirement.

---

## Tier 2 — Ranking quality (encodes explicit JD requirements)

**`ranker/config.py`**
- Added `CV_KEYWORDS`, `SPEECH_KEYWORDS`, `ROBOTICS_KEYWORDS`, `NLP_IR_KEYWORDS` (deliberately
  excluding short, collision-prone substrings like `gan`/`visual`/`ros`/`search`/`rag`).
- Expanded `SERVICE_FIRMS` with BPO/managed-services shops (Genpact, WNS, EXL, Firstsource, …).
- Added graded-penalty bounds (`CV_MISMATCH_*`), a precise CV-title override
  (`CV_TITLE_TERMS`, `CV_TITLE_NLP_EXEMPTION_HITS`), and title-chaser thresholds.

**`ranker/features.py`** (`compute_risk_features`)
- `domain_mismatch_score`: CV/speech/robotics evidence vs NLP/IR evidence, read from the
  **verifiable work** (headline, summary, career descriptions + titles) — **not** the skills array,
  which is where keyword-stuffing lives (per the challenge pointers). An explicit CV/speech/robotics
  *current title* forces a strong mismatch **unless** the candidate also shows significant NLP/IR
  work (the JD's "*without* NLP/IR exposure" carve-out).
- `service_ratio` (graded service-firm tenure share), `avg_tenure_months`, `is_title_chaser`.
- All exposed as `risk_*` features so the LTR model can also use them (`risk_nlp_ir_keyword_hits`
  is now a top model feature).

**`ranker/scorer.py`** — added shared, graded multiplier helpers used by *both* scoring paths:
`compute_cv_mismatch_multiplier`, `compute_consulting_multiplier` (graded, replaces the binary
all-service gate), `compute_title_chaser_multiplier`. Wired into `score_candidate`.

**`rank.py`** — applied the same graded multipliers in the LTR batch path and the heuristic
fast path, keeping the two consistent.

**`scripts/train_ltr.py`**
- Labels now mark severely off-domain profiles (`domain_mismatch > 0.65`) as relevance 0.
- Replaced random 500-item query groups with **few large (~8k) representative groups**
  (LightGBM caps a query at 10k rows, so a single group is not possible). After a global shuffle
  each group is a representative tier mix, giving far more meaningful pairwise comparisons than 500.
- Added an optional `--data <dir>` argument.

---

## Tier 3 — Reasoning quality (Stage-4 manual review)

**`ranker/reasoning.py`**
- Calibrated the ELITE templates (removed "ML veteran / premium pedigree / giants" framing).
- Added a tone guard: clearly-junior current titles do not receive the most emphatic wording.
- Reworded the experience-band template so it no longer calls a low-YOE candidate "Senior".
- Result: 100/100 unique reasoning strings, tone consistent with rank, honest concerns surfaced.

**`scripts/generate_resumes.py`** — added an optional `--candidates` argument; regenerated all
100 resumes so `output/top_100_resumes/` matches the new ranking.

---

## Regenerated artifacts
`input/features.parquet`, `input/ltr_model.txt`, `input/feature_names.json`,
`input/candidate_ids.json`, `output/submission.csv`, `output/top_100_resumes/*.md`.
Removed the stale `output/test_submission.csv` (an old 50-candidate sample run).

---

## Still TODO before submitting (cannot be done from code)
1. Fill `phone`, the second member's email, and a real `sandbox_link` in `submission_metadata.yaml`.
2. Regenerate or remove the stale `pola_verification_report.md` / `profile_verification_report.md`
   (they still describe the *previous* top candidates and now contradict `output/submission.csv`).
3. Push to the correct GitHub repo with a real commit sequence.

## Known limitation
Synthetic labels are skewed (≈74.6% tier-0; only 45 tier-3) and derived from the same features
the model sees, so the LTR is bounded by label quality. The graded penalties above are the
high-confidence wins; deeper de-circularization would require ground-truth labels (not provided).
