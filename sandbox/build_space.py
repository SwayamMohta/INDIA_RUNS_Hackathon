"""
build_space.py — Assemble a self-contained HuggingFace Space from the repo.

The Space must be inference-only and standalone, so it ships ONLY what the
`rank.py --standalone` path needs: the app, the ranker package, rank.py, and the
trained-model artifacts (NOT the 120 MB BM25 pickle or the 100k feature parquet).

Usage:
    python sandbox/build_space.py            # writes sandbox/_space/

Then deploy (see SUBMISSION_GUIDE.md for the full walk-through):
    1. Create a Gradio Space at https://huggingface.co/new-space
    2. cd sandbox/_space && git init && git remote add origin <space-git-url>
    3. git add . && git commit -m "Redrob ranker sandbox" && git push -u origin main
"""

import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = os.path.join(HERE, "_space")

# (src_relative_to_repo, dst_relative_to_OUT)
FILES = [
    ("sandbox/app.py", "app.py"),
    ("sandbox/README.md", "README.md"),
    ("sandbox/requirements.txt", "requirements.txt"),
    ("rank.py", "rank.py"),
    ("input/ltr_model.txt", "input/ltr_model.txt"),
    ("input/feature_names.json", "input/feature_names.json"),
    ("input/sample_candidates.json", "input/sample_candidates.json"),
]


def main():
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)

    # Copy the ranker package (excluding caches).
    shutil.copytree(
        os.path.join(REPO, "ranker"),
        os.path.join(OUT, "ranker"),
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )

    for src, dst in FILES:
        s = os.path.join(REPO, src)
        d = os.path.join(OUT, dst)
        os.makedirs(os.path.dirname(d), exist_ok=True)
        if not os.path.exists(s):
            raise FileNotFoundError(f"Missing required file: {s}")
        shutil.copy2(s, d)

    # Sanity: the trained model must be present (this is the only "trained" artifact).
    assert os.path.exists(os.path.join(OUT, "input", "ltr_model.txt")), "ltr_model.txt missing"

    total = sum(
        os.path.getsize(os.path.join(root, f))
        for root, _, files in os.walk(OUT) for f in files
    )
    print(f"[build_space] Assembled Space at: {OUT}")
    print(f"[build_space] Total size: {total/1e6:.2f} MB (inference-only — no parquet/pickle)")
    print("[build_space] Contents:")
    for root, _, files in os.walk(OUT):
        for f in sorted(files):
            rel = os.path.relpath(os.path.join(root, f), OUT)
            print(f"    {rel}")
    print("\n[build_space] Next: create a Gradio Space, then from sandbox/_space:")
    print("    git init && git add . && git commit -m 'Redrob ranker sandbox'")
    print("    git remote add origin https://huggingface.co/spaces/<user>/<space>")
    print("    git push -u origin main")


if __name__ == "__main__":
    main()
