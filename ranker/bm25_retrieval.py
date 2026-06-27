"""
bm25_retrieval.py — Stage 2 coarse retrieval using BM25.

Indexes candidate career_history descriptions (not summaries — too noisy)
and retrieves top-K candidates most similar to the JD query.

Why career_history descriptions and NOT summaries?
- Summaries are self-reported and susceptible to keyword inflation
- Career descriptions contain verifiable, concrete work evidence
- BM25 on descriptions rewards actual usage of technical terms in context
"""

import pickle
import os
from typing import List, Dict, Any, Tuple

from rank_bm25 import BM25Okapi
from tqdm import tqdm

from ranker.config import JD_QUERY, BM25_TOP_K


# Common English stopwords to remove for better BM25 matching
_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "need", "dare",
    "it", "its", "i", "me", "my", "we", "our", "you", "your", "he", "she",
    "his", "her", "they", "their", "them", "this", "that", "these", "those",
    "as", "if", "then", "than", "so", "not", "no", "nor", "too", "very",
    "also", "just", "about", "above", "after", "before", "between", "into",
    "through", "during", "out", "up", "down", "over", "under", "again",
    "each", "all", "both", "few", "more", "most", "other", "some", "such",
    "only", "own", "same", "here", "there", "when", "where", "why", "how",
    "what", "which", "who", "whom", "while",
}

# Compound terms to keep together (replace spaces with underscores)
_COMPOUND_TERMS = [
    "learning_to_rank", "vector_search", "vector_database",
    "semantic_search", "dense_retrieval", "approximate_nearest_neighbor",
    "information_retrieval", "large_language_model", "retrieval_augmented_generation",
    "natural_language_processing", "machine_learning", "deep_learning",
    "model_serving", "model_deployment", "feature_store",
    "hugging_face", "fine_tuning", "instruction_tuning",
    "ab_testing", "online_evaluation", "neural_network",
    "recommendation_system", "search_engine",
]


def _tokenize(text: str) -> List[str]:
    """Tokenizer with stopword removal and compound-term preservation."""
    text = text.lower()
    # Replace compound terms with underscored versions
    for ct in _COMPOUND_TERMS:
        spaced = ct.replace("_", " ")
        if spaced in text:
            text = text.replace(spaced, ct)
    tokens = text.split()
    # Remove stopwords and very short tokens
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


def build_bm25_index(
    candidates: List[Dict[str, Any]],
    verbose: bool = True
) -> Tuple[BM25Okapi, List[str]]:
    """
    Build BM25 index from candidate career_history descriptions.

    Args:
        candidates: List of candidate dicts
        verbose: Show progress bar

    Returns:
        (bm25_model, candidate_ids) — ordered list matching the BM25 corpus
    """
    corpus = []
    candidate_ids = []

    iterator = tqdm(candidates, desc="Building BM25 index") if verbose else candidates

    for cand in iterator:
        cid = cand.get("candidate_id", "")
        career = cand.get("career_history", [])
        profile = cand.get("profile", {})

        # Concatenate career descriptions + headline (not full summary)
        career_text = " ".join(
            role.get("description", "") for role in career
        )
        headline = profile.get("headline", "")
        # Add title from each role for context
        role_titles = " ".join(role.get("title", "") for role in career)

        combined = f"{headline} {role_titles} {career_text}"
        tokens = _tokenize(combined)

        corpus.append(tokens)
        candidate_ids.append(cid)

    print(f"[bm25] Building index over {len(corpus)} candidates...")
    bm25 = BM25Okapi(corpus)
    print("[bm25] Index built.")

    return bm25, candidate_ids


def query_bm25(
    bm25: BM25Okapi,
    candidate_ids: List[str],
    query: str = JD_QUERY,
    top_k: int = BM25_TOP_K,
) -> List[Tuple[str, float]]:
    """
    Query BM25 index and return top-K candidates.

    Args:
        bm25: Trained BM25 model
        candidate_ids: Ordered list of candidate IDs (parallel to corpus)
        query: Query string (default: JD_QUERY from config)
        top_k: Number of candidates to retrieve

    Returns:
        List of (candidate_id, normalized_bm25_score) tuples, descending
    """
    query_tokens = _tokenize(query)
    scores = bm25.get_scores(query_tokens)

    # Get top-k indices
    import numpy as np
    top_indices = np.argsort(scores)[::-1][:top_k]

    # Normalize scores to [0, 1]
    max_score = scores[top_indices[0]] if len(top_indices) > 0 else 1.0
    if max_score == 0:
        max_score = 1.0

    results = []
    for idx in top_indices:
        normalized = float(scores[idx]) / max_score
        results.append((candidate_ids[idx], normalized))

    return results


def save_bm25_artifacts(
    bm25: BM25Okapi,
    candidate_ids: List[str],
    output_dir: str = "data"
):
    """Save BM25 index and candidate ID list to disk."""
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "bm25_index.pkl"), "wb") as f:
        pickle.dump({"bm25": bm25, "candidate_ids": candidate_ids}, f)
    print(f"[bm25] Saved index to {output_dir}/bm25_index.pkl")


def load_bm25_artifacts(data_dir: str = "data") -> Tuple[BM25Okapi, List[str]]:
    """Load BM25 index from disk."""
    path = os.path.join(data_dir, "bm25_index.pkl")
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj["bm25"], obj["candidate_ids"]
