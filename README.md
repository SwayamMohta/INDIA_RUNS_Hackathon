# Redrob Intelligent Candidate Discovery & Ranking System

Ranks the top-100 candidates from a 100,000-candidate pool against the *Senior AI
Engineer* job description, producing `output/submission.csv` (rank, score, grounded
reasoning) within the hackathon's compute budget (**≤5 min, ≤16 GB, CPU-only, no
network** during ranking).

## Approach (one paragraph)

A two-stage retrieve-then-rank pipeline. **BM25** over *verifiable* career-history
descriptions (not the self-reported skills array, where keyword-stuffing lives) gives a
coarse relevance score. A **LightGBM LambdaMART** learning-to-rank model then scores every
candidate over ~80 engineered features (career trajectory, product-vs-service ratio,
domain-shipping evidence, JD skill/title overlap, behavioural availability). On top of the
model score we apply **graded multiplicative penalties** that encode the JD's *explicit*
anti-signals: a domain-mismatch penalty down-weights computer-vision/speech/robotics-primary
profiles that lack NLP/IR evidence, and a graded consulting penalty scales with
service-firm tenure share. **Honeypots** (impossible profiles — e.g. tenure predating a
company's founding) are detected against real company founding years and hard-dropped, and
**keyword-stuffers** (AI skills listed but absent from actual work history) are penalised.
Reasoning is template-based and grounded only in verifiable profile fields — **no LLM calls
during ranking**.

## Repository layout

| Path | What it is |
|---|---|
| `rank.py` | **The ranking step** — produces `submission.csv`. The ≤5-min, CPU-only, no-network step. |
| `precompute.py` | Offline feature/BM25 builder for the full pool (no time limit). |
| `scripts/train_ltr.py` | Offline LambdaMART **training** — produces `input/ltr_model.txt`. |
| `ranker/` | Core package: feature engineering, scoring, honeypot/stuffer detection, reasoning, BM25. |
| `input/` | Trained model (`ltr_model.txt`), feature list, precomputed `features.parquet`, candidate ids. |
| `output/` | `submission.csv` (the final ranking) + `top_100_resumes/` (per-candidate profile dumps). |
| `sandbox/` | HuggingFace Space (Gradio) for the §10.5 demo + `build_space.py` assembler. |
| `*_verification_report.md` | Grounded, candidate-by-candidate audit of the ranking (Stage-4 evidence). |
| `requirements.txt` | Dependency spec. |
| `submission_metadata.yaml` | Portal submission metadata. |

## Inference vs. training — what the judges & sandbox run

The system has three clearly-separated stages. **Reproduction and the sandbox run ONLY
the inference stage.** The two offline stages were run once by us; their outputs are
committed to the repo.

| Stage | Script | When | Committed output | Run at reproduction? |
|---|---|---|---|---|
| **Training** | `scripts/train_ltr.py` | once, offline | `input/ltr_model.txt`, `input/feature_names.json` | **No** |
| **Precompute** | `precompute.py` | once, offline | `input/features.parquet` (+ regenerable `bm25_index.pkl`) | No (optional) |
| **Inference / ranking** | `rank.py` | every run | `output/submission.csv` | **Yes** |

The trained model ships in the repo (`input/ltr_model.txt`, 0.3 MB). Nothing retrains it.

## Setup

```bash
pip install -r requirements.txt
# Place the organizer-provided pool at input/candidates.jsonl (git-ignored; supplied at reproduction time).
```

## Reproduce the submission (single command)

`features.parquet` and `ltr_model.txt` are committed, so the full top-100 is reproduced by
the ranking step alone (the BM25 index, the only large/git-ignored artifact, is rebuilt
from `--candidates` automatically and cached):

```bash
python rank.py --candidates input/candidates.jsonl --data input --output output/submission.csv
```

Verified end-to-end on a 16 GB CPU machine: **well under a minute** (≈15–55 s depending on
machine/load and whether `input/bm25_index.pkl` is cached vs rebuilt from scratch) — comfortably
within the 5-minute budget, and reproduces the committed `output/submission.csv` byte-for-byte.

> Regenerating the offline artifacts (not needed for reproduction):
> `python precompute.py --candidates input/candidates.jsonl --output input` then
> `python scripts/train_ltr.py --data input`.

## Sandbox (submission_spec.md §10.5)

**Hosted demo:** https://huggingface.co/spaces/suhas9545/redrob-ranker — a HuggingFace Space (Gradio). Upload a ≤100-candidate
sample (JSON array or JSONL) or use the bundled 50-candidate sample; it ranks them
end-to-end and returns a CSV. It is **inference-only**: it loads the trained model and never
trains/precomputes. Assemble & deploy it with `python sandbox/build_space.py` (see
`SUBMISSION_GUIDE.md`).

**Run the same inference-only path locally on a small sample:**
```bash
python rank.py --standalone --sample --data input --output sample_submission.csv
# or on your own ≤100 file:
python rank.py --standalone --candidates my_sample.json --data input --output out.csv
```
`--standalone` computes features on the fly for the given sample and applies the trained
model — no precompute, no training.

**Docker fallback (§10.5):** an inference-only image is provided.
```bash
docker build -t redrob-ranker .
docker run --rm -v "$PWD:/work" redrob-ranker \
  --candidates input/candidates.jsonl --data input --output output/submission.csv
```

## Validate

```bash
python scripts/validate_submission.py output/submission.csv   # -> "Submission is valid."
```

## Traps & honeypots

The dataset embeds keyword-stuffers, off-domain profiles, and ~80 honeypots (forced to
tier 0; >10% in the top-100 is a disqualifier). The system avoids these by reasoning over
*verifiable work evidence* rather than the skills array, hard-dropping detectable
honeypots, and penalising stuffers/off-domain/consulting-heavy profiles. The committed
ranking contains **0 honeypots** in the top-100.
