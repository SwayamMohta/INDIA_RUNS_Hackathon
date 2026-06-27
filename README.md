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

Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

---

## How to Run (Reproduction Command)

To run the ranking pipeline on the full dataset and generate the final output, run the following command from the `FINAL/` directory:

```bash
python rank.py --candidates input/candidates.jsonl --data input --output output/submission.csv
```

### Script Arguments:
- `--candidates`: Path to the input JSONL file containing candidate profiles (default: `candidates.jsonl`).
- `--data`: Path to the directory containing precomputed indices, feature parquet, and models (default: `data`).
- `--output` / `--out`: Path to write the final CSV output (default: `submission.csv`).

The pipeline will finish in **under 20 seconds** on a single CPU core, filtering out honeypots and producing a fully compliant top-100 list with fact-grounded reasoning.

---

## Verifying Compliance

To check that the generated CSV strictly adheres to the submission rules (ranks 1-100, exactly 100 candidates, etc.):

```bash
python scripts/validate_submission.py output/submission.csv
```
This will print `Submission is valid.` upon success.
