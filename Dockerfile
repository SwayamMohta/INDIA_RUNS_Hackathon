# Inference-only image for the Redrob ranker (submission_spec.md §10.5 / Stage-3).
# The ENTRYPOINT is the ranking step (rank.py). Training (scripts/train_ltr.py) and full
# precompute are NOT part of this image — the trained model ships in input/ltr_model.txt.
FROM python:3.10-slim

WORKDIR /work

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the code + committed artifacts (.dockerignore excludes the pool, .git, caches).
COPY . .

# Pass --candidates / --data / --output as `docker run` args; defaults reproduce the full
# submission when input/candidates.jsonl is present (mount it with -v or COPY at build).
ENTRYPOINT ["python", "rank.py"]
CMD ["--candidates", "input/candidates.jsonl", "--data", "input", "--output", "output/submission.csv"]
