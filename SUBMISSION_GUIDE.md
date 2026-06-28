# Redrob Hackathon — Final Submission Guide

This is the **single checklist** to take the repo from its current state to a perfect
submission. Work top to bottom. Items marked **[ACTION]** are the only things that cannot
be done from code and need you.

---

## 0. Current state (already done)

- ✅ Full ranking system: BM25 → LightGBM LambdaMART LTR → graded JD-penalties → grounded reasoning.
- ✅ `output/submission.csv` — 100 rows, validator-clean, 0 honeypots in top-100.
- ✅ Reproduces **byte-for-byte** via a single command (verified well under a minute on a 16 GB CPU box).
- ✅ **Inference-only** standalone path (`rank.py --standalone`) — runs the trained model on a
  ≤100 sample with no training/precompute.
- ✅ HuggingFace **sandbox** built under `sandbox/` (Gradio, inference-only).
- ✅ `Dockerfile` (inference-only) for the §10.5 Docker fallback.
- ✅ Docs: `README.md`, `CHANGES.md`, verification reports, this guide.
- ✅ Repo moved into the project root; `submission` branch created; dev clutter removed.

---

## 1. [ACTION] Fill the three blanks in `submission_metadata.yaml`

These are placeholders right now:

| Field | Current value | What to put |
|---|---|---|
| `primary_contact.phone` | `+91-XXXXXXXXXX` | A real reachable phone (top-N outreach). |
| `team_members[1].email` | `FILL_ME@example.com` (Sri Suhas P) | The teammate's real email. |
| `sandbox_link` | `FILL_ME_HF_SPACE_URL` | The HF Space URL from step 2. |

Also confirm `github_repo` (`https://github.com/SwayamMohta/INDIA_RUNS_Hackathon`) and the
team/contact names are correct.

---

## 2. [ACTION] Deploy the HuggingFace Space (the §10.5 sandbox)

The sandbox loads the trained model and ranks a ≤100 sample — **inference only**.

```bash
# (a) Assemble a clean, self-contained Space (app + ranker + rank.py + trained model; no parquet/pickle)
python sandbox/build_space.py          # writes sandbox/_space/  (~0.76 MB)

# (b) Create a Space:  https://huggingface.co/new-space  → SDK: Gradio  → name it e.g. redrob-ranker
#     (HF will accept the committed README.md header: sdk_version 4.44.1, app_file app.py)

# (c) Push the assembled Space:
cd sandbox/_space
git init && git add . && git commit -m "Redrob ranker sandbox (inference-only)"
git remote add origin https://huggingface.co/spaces/<your-hf-user>/redrob-ranker
git push -u origin main
```

If pushing over HTTPS asks for a password, use a **HF access token** (Settings → Access
Tokens, role *write*) as the password. Once the Space builds, open it, click **Rank** to
confirm it returns a CSV, then paste the Space URL into `sandbox_link` (step 1) and into
`README.md` (`<SANDBOX_LINK>`).

> No HF account / Space build problems? The `Dockerfile` is an accepted §10.5 fallback —
> see README "Docker fallback". You can submit the `docker run` recipe instead of a hosted link.

---

## 3. Pre-flight verification (run before pushing)

```bash
# Reproduce the full submission and confirm it matches the committed CSV:
python rank.py --candidates input/candidates.jsonl --data input --output /tmp/check.csv
python scripts/validate_submission.py /tmp/check.csv        # -> "Submission is valid."
diff /tmp/check.csv output/submission.csv && echo "MATCH"   # (no output from diff = identical)

# Confirm the inference-only sandbox path works:
python rank.py --standalone --sample --data input --output /tmp/sample.csv
```

(On Windows PowerShell use a real temp path, e.g. `$env:TEMP\check.csv`.)

---

## 4. [ACTION] Push the `submission` branch to GitHub

You are on branch `submission` (it contains the full 5-commit history plus the Round-2 commit).

```bash
git push -u origin submission
```

If HTTPS prompts for credentials, use a **GitHub Personal Access Token** (Settings →
Developer settings → Tokens, `repo` scope) as the password. The repo owner is
`SwayamMohta`; you (srisuhas) need collaborator write access.

---

## 5. [ACTION] Finalise: make `submission` the new `main`, delete the rest

Do this once you're happy with `submission`. It rewrites `main` to the submission content
and removes the intermediate branches.

```bash
# --- Local ---
git checkout submission
git branch -m submission main                              # rename submission -> main
git branch -D "improvements/reproducibility-and-ranking"   # delete the intermediate branch
# (the old local 'main' was renamed away; if a stale one remains: git branch -D oldmain)

# --- Remote ---
git push origin main --force                               # publish the new main
git push origin --delete submission 2>/dev/null || true                 # remove the pushed submission branch
git push origin --delete "improvements/reproducibility-and-ranking" 2>/dev/null || true  # if it was ever pushed
git remote set-head origin -a
```

After this, `main` is the only branch and it is the perfect submission. Verify on GitHub
that `main` shows `rank.py`, `README.md`, `output/submission.csv`, `sandbox/`, and the
commit history (Stage-4 checks for real iteration, not a single dump — this history has it).

> Grant the organizers read access at Stage 3 if the repo is private (the email is given then).

---

## 6. [ACTION] Submit on the portal (§10)

Have ready and upload:

- **CSV:** `output/submission.csv` — rename to your **registered participant ID** (e.g. `team_xxx.csv`).
  Re-run the validator on the renamed file first.
- **Metadata (§10.2):** team name, primary contact (name/email/**phone**), every member's
  name+email, GitHub URL, **sandbox link**, AI-tools declaration, compute summary, methodology
  (≤200 words — already in `submission_metadata.yaml`).
- **GitHub repo:** reachable, with `submission_metadata.yaml` at root (it is).
- 3 submissions max; the **last valid** one counts.

---

## 7. How the submission maps to the evaluation stages

| Stage | Requirement | How this repo satisfies it |
|---|---|---|
| 1 — Format | Exactly 100 rows, unique ranks, non-increasing score, valid IDs | `output/submission.csv` passes `validate_submission.py`. |
| 2 — Scoring | NDCG@10/50, MAP, P@10 vs hidden truth | LTR-ranked top-100 with JD-aligned penalties; top picks are product-company ML/AI engineers. |
| 3 — Reproduction + honeypots | Re-run ranking in ≤5 min/16 GB/CPU/no-net; <10% honeypots | Single command, ~17–42 s; **0 honeypots** in top-100; trained model committed (no retrain needed). |
| 4 — Manual review | Reasoning quality, methodology, real git history, code quality | 100 unique grounded reasonings; `CHANGES.md` + verification reports; multi-commit history; clean package. |
| 5 — Interview | Defend architecture | `README.md` "Approach", the inference-vs-training table, and `ranker/` modules each map to a design choice. |

---

## 8. Key facts to know for the interview / defense

- **Why BM25 over career descriptions, not the skills array:** the skills array is where
  keyword-stuffing lives; verifiable work history is harder to fake (`ranker/bm25_retrieval.py`).
- **Why graded penalties, not hard filters:** the JD rejects CV/speech/robotics-*primary*
  profiles *without* NLP/IR — a graded `domain_mismatch` multiplier with an NLP/IR carve-out,
  not a blanket ban (`ranker/scorer.py`, `ranker/config.py`).
- **Honeypots:** checked against real company founding years and other impossibilities, then
  hard-dropped (`ranker/honeypot.py`).
- **No LLM at inference:** reasoning is template-based and grounded only in profile fields —
  this is *why* it scales to 100k in seconds on CPU (`ranker/reasoning.py`).
- **Inference vs training:** `train_ltr.py` ran once offline → `ltr_model.txt` (committed);
  reproduction/sandbox run only `rank.py`. See the table in `README.md`.
- **Known limitation:** synthetic training labels are skewed/derived from the same features,
  so the LTR is bounded by label quality (`CHANGES.md`).
