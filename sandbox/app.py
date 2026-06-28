"""
app.py — Redrob Ranker sandbox (HuggingFace Space, Gradio).

INFERENCE ONLY. This app loads the *already-trained* LightGBM LambdaMART model
(`input/ltr_model.txt`) and ranks a small candidate sample (<=100) end-to-end. It
never runs `train_ltr.py` (training) or a full `precompute.py` over the 100k pool.
Under the hood it calls the exact documented command:

    python rank.py --standalone --candidates <sample> --data input --output <csv>

which computes features on the fly for the uploaded sample and applies the trained
model — the same code path that produced the submission, restricted to the sample.

Satisfies submission_spec.md §10.5: accepts a <=100 candidate sample, runs the ranker
end-to-end on CPU within the compute budget, and returns a ranked CSV.
"""

import json
import os
import subprocess
import sys
import tempfile

import gradio as gr
import pandas as pd

# ── Locate the repo (works whether app.py is at repo root, in sandbox/, or in an
#    assembled HF Space where rank.py sits beside app.py). ──────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = HERE if os.path.exists(os.path.join(HERE, "rank.py")) else os.path.dirname(HERE)
RANK_PY = os.path.join(REPO, "rank.py")
DATA_DIR = os.path.join(REPO, "input")          # holds ltr_model.txt + feature_names.json
SAMPLE = os.path.join(DATA_DIR, "sample_candidates.json")

MAX_CANDIDATES = 100                              # §10.5 sandbox cap
TIMEOUT_S = 300                                   # 5-minute compute budget


def _read_candidates(path: str) -> list:
    """Read a candidate list from a JSON array, a single JSON object, or JSONL."""
    if path.endswith(".jsonl"):
        out = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def rank_sample(uploaded_file):
    """Rank the uploaded sample (or the bundled 50-candidate sample) and return
    (preview_dataframe, downloadable_csv_path, run_log)."""
    log_lines = []

    # 1) Resolve input: uploaded file, else the bundled sample. gr.File passes a string
    #    path (type="filepath"); older gradio passes an object with .name — handle both.
    if uploaded_file is None:
        src = SAMPLE
    else:
        src = uploaded_file if isinstance(uploaded_file, str) else uploaded_file.name
    if not os.path.exists(src):
        return None, None, f"ERROR: input not found: {src}"

    try:
        candidates = _read_candidates(src)
    except Exception as e:  # noqa: BLE001
        return None, None, f"ERROR: could not parse input as JSON array / JSONL: {e}"

    if not candidates:
        return None, None, "ERROR: no candidates found in the input file."

    note = ""
    if len(candidates) > MAX_CANDIDATES:
        candidates = candidates[:MAX_CANDIDATES]
        note = f"(input truncated to the first {MAX_CANDIDATES} candidates per §10.5)\n"
    log_lines.append(f"Loaded {len(candidates)} candidate(s) {note}".strip())

    # 2) Normalise to a temp JSON array and run the documented standalone command.
    workdir = tempfile.mkdtemp(prefix="redrob_sandbox_")
    in_path = os.path.join(workdir, "sample.json")
    out_path = os.path.join(workdir, "ranked.csv")
    with open(in_path, "w", encoding="utf-8") as f:
        json.dump(candidates, f)

    cmd = [sys.executable, RANK_PY, "--standalone",
           "--candidates", in_path, "--data", DATA_DIR, "--output", out_path]
    log_lines.append("Running (INFERENCE ONLY — trained model, no training/precompute):")
    log_lines.append("  " + " ".join(["python", "rank.py", "--standalone",
                                       "--candidates", "<sample>", "--data", "input",
                                       "--output", "<csv>"]))
    try:
        proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return None, None, "\n".join(log_lines) + f"\n\nERROR: exceeded {TIMEOUT_S}s budget."

    log_lines.append("")
    log_lines.append(proc.stdout.strip()[-2000:])
    if proc.returncode != 0:
        log_lines.append("\nSTDERR:\n" + proc.stderr.strip()[-2000:])
        return None, None, "\n".join(log_lines)

    if not os.path.exists(out_path):
        return None, None, "\n".join(log_lines) + "\n\nERROR: no output CSV produced."

    df = pd.read_csv(out_path)
    log_lines.append(f"\nDone — produced a ranked CSV with {len(df)} row(s).")
    return df, out_path, "\n".join(log_lines)


with gr.Blocks(title="Redrob Intelligent Candidate Ranker — Sandbox") as demo:
    gr.Markdown(
        """
        # Redrob Intelligent Candidate Ranker — Sandbox
        **Inference-only demo.** Upload a small candidate sample (**≤100**, JSON array or
        JSONL matching `candidate_schema.json`), or just click **Rank** to use the bundled
        50-candidate sample.

        This loads the **already-trained** LightGBM LambdaMART model and ranks the sample
        end-to-end on CPU — it never runs training or full precomputation. It calls the
        exact documented command `python rank.py --standalone ...`.
        """
    )
    with gr.Row():
        file_in = gr.File(label="Candidate sample (.json array or .jsonl, ≤100). Leave empty to use the bundled sample.",
                          file_types=[".json", ".jsonl"])
    run_btn = gr.Button("Rank candidates", variant="primary")
    out_df = gr.Dataframe(label="Ranking (rank, candidate_id, score, reasoning)", wrap=True)
    out_csv = gr.File(label="Download ranked CSV")
    out_log = gr.Textbox(label="Run log (proves inference-only, no network)", lines=16)

    run_btn.click(rank_sample, inputs=[file_in], outputs=[out_df, out_csv, out_log])

if __name__ == "__main__":
    demo.launch()
