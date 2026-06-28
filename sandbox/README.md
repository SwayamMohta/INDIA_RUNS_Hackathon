---
title: Redrob Candidate Ranker Sandbox
emoji: 🎯
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
---

# Redrob Intelligent Candidate Ranker — Sandbox

**Inference-only demo** for the Redrob hackathon (submission_spec.md §10.5).

Upload a small candidate sample (**≤100**, a JSON array or JSONL matching
`candidate_schema.json`) or click **Rank** to use the bundled 50-candidate sample.
The Space loads the **already-trained** LightGBM LambdaMART model
(`input/ltr_model.txt`) and ranks the sample end-to-end on CPU.

It calls the exact documented command:

```bash
python rank.py --standalone --candidates <sample> --data input --output <csv>
```

which computes features on the fly for the provided sample and applies the trained
model — the same code path that produced the competition submission, restricted to the
sample. **No training (`train_ltr.py`) and no full precomputation run here.** No network
calls during ranking.

This Space is assembled from the project repo by `sandbox/build_space.py`. The full
ranking system, training, and reproduction instructions live in the GitHub repo's
`README.md`.
