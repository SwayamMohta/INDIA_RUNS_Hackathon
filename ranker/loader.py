"""
loader.py — Streaming JSONL parser for the 487MB candidates file.

Reads candidates line-by-line to avoid loading the entire file into memory.
Also supports reading from the sample JSON file for testing.
"""

import json
import os
from typing import Generator, Dict, Any


def stream_candidates(filepath: str) -> Generator[Dict[str, Any], None, None]:
    """
    Stream candidates from a JSONL file (one JSON object per line).
    Skips malformed lines with a warning.

    Args:
        filepath: Path to candidates.jsonl

    Yields:
        dict: Parsed candidate record
    """
    line_num = 0
    errors = 0
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line_num += 1
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                errors += 1
                if errors <= 5:  # only warn first 5 errors
                    print(f"[loader] WARNING: skipped malformed line {line_num}: {e}")


def load_sample_candidates(filepath: str) -> list:
    """
    Load the sample_candidates.json (a JSON array, not JSONL).
    Used for testing and development.

    Args:
        filepath: Path to sample_candidates.json

    Returns:
        list of candidate dicts
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def count_candidates(filepath: str) -> int:
    """Count lines in the JSONL file (fast estimate of candidate count)."""
    count = 0
    with open(filepath, "rb") as f:
        for _ in f:
            count += 1
    return count
