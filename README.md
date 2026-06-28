# Redrob Candidate Discovery & Ranking System (FINAL Deliverables)

This directory contains the production-grade candidate discovery and ranking system. It has been refactored and organized into a clean, reproducible structure for final submission.

## Directory Structure
- `rank.py` - Main pipeline run-script.
- `precompute.py` - Offline feature builder and BM25 precomputation.
- `ranker/` - Core ranking Python package containing the scoring logic, feature definitions, and honeypot detection.
- `input/` - Raw candidate data (`candidates.jsonl`), LightGBM model weights, and precomputed features (`bm25_index.pkl`, `features.parquet`).
- `output/` - Final validated submission output (`submission.csv`).
- `scripts/` - Offline training script (`train_ltr.py`), validation script (`validate_submission.py`), and helper utility scripts.
- `requirements.txt` / `pyproject.toml` - Project dependency specifications.
- `submission_metadata.yaml` - Portal submission metadata.
- `pola_verification_report.md` / `profile_verification_report.md` - Analysis, manual deep-dives, and quality validation records.

---

## Installation & Setup

Ensure all dependencies are installed (LightGBM is required — it powers the LTR ranker):
```bash
pip install -r requirements.txt
```

Place the organizer-provided candidate pool at `input/candidates.jsonl` (this file is
git-ignored because of its size; it is supplied at reproduction time).

---

## How to Run (Reproduction)

The system has a one-time **precompute** step (no time limit) and a fast **ranking** step
(the < 5-minute, CPU-only, no-network step that produces the submission CSV).

### Step 1 — Precompute (one-time, ~10-15 min for 100k candidates)
Builds the BM25 index (`input/bm25_index.pkl`) and the feature matrix
(`input/features.parquet`). A pre-built `features.parquet` and trained `ltr_model.txt`
ship in the repo; this step regenerates `bm25_index.pkl` (and refreshes features):
```bash
python precompute.py --candidates input/candidates.jsonl --output input
```
(Optional — retrain the LTR model after a feature change: `python scripts/train_ltr.py`)

### Step 2 — Rank (the < 5-minute step)
```bash
python rank.py --candidates input/candidates.jsonl --data input --output output/submission.csv
```
This loads the precomputed artifacts + the LightGBM LTR model and finishes in **under
30 seconds** on a single CPU core. If `bm25_index.pkl` is absent it is rebuilt on the
fly from `--candidates` (slower, but the LTR model is still used). The ranking step makes
no network calls and uses no GPU.

### Script Arguments:
- `--candidates`: Path to the input JSONL file containing candidate profiles.
- `--data`: Directory containing precomputed indices, feature parquet, and the LTR model (default: `input`).
- `--output` / `--out`: Path to write the final CSV output.

---

## Sandbox / Docker (for §10.5 reproduction)

If the hosted sandbox link is unavailable, the system reproduces unmodified via Docker:
```bash
docker run --rm -v "$PWD:/work" -w /work python:3.10-slim bash -c \
  "pip install -r requirements.txt && \
   python precompute.py --candidates input/candidates.jsonl --output input && \
   python rank.py --candidates input/candidates.jsonl --data input --output output/submission.csv"
```
For the small-sample sanity check, pass `--sample` to operate on `sample_candidates.json`.

---

## Verifying Compliance

To check that the generated CSV strictly adheres to the submission rules (ranks 1-100, exactly 100 candidates, etc.):

```bash
python scripts/validate_submission.py output/submission.csv
```
This will print `Submission is valid.` upon success.
