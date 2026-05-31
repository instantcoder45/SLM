"""
Lecture Topic Registry.

Loads keyword metadata from all lecture slide decks in slides_db at startup
and provides lecture-aware filtering for RAG retrieval.

Each lecture's meta.pkl contains TF-IDF keywords extracted during the
RefineSlides indexing step.  This module embeds those keyword signatures
and exposes a fast cosine-similarity lookup so the RAG tool can focus
its FAISS search on the most relevant lectures.
"""

from __future__ import annotations

import os
import pickle
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Module-level state (populated by init_lecture_registry)
# ---------------------------------------------------------------------------
_lecture_keywords: dict[str, list[str]] = {}   # {"Lecture-1": ["architecture", "cache", ...], ...}
_lecture_embeddings: dict[str, np.ndarray] = {}  # {"Lecture-1": np.array([...]), ...}
_embedder = None


def init_lecture_registry(config) -> None:
    """
    Load keyword metadata from every lecture folder in slides_db and
    pre-compute keyword-signature embeddings for fast lookup.

    Args:
        config: An agents.config.Config instance with ``chapters_db_path``
                and ``embed_model`` attributes.
    """
    global _lecture_keywords, _lecture_embeddings, _embedder
    from sentence_transformers import SentenceTransformer

    db_root = config.chapters_db_path
    if not os.path.exists(db_root):
        print(f"⚠️  Lecture registry: DB path not found: {db_root}")
        return

    # Reuse the same embedding model the RAG tool uses
    if _embedder is None:
        _embedder = SentenceTransformer(config.embed_model)

    loaded = 0
    for entry in sorted(os.listdir(db_root)):
        meta_path = os.path.join(db_root, entry, "meta.pkl")
        if not os.path.exists(meta_path):
            continue

        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        keywords = meta.get("keywords", [])
        if not keywords:
            continue

        _lecture_keywords[entry] = keywords

        # Create a single embedding from the joined keyword string
        signature_text = " ".join(keywords)
        _lecture_embeddings[entry] = _embedder.encode(signature_text)
        loaded += 1

    print(f"✅ Lecture registry: Loaded keyword signatures for {loaded} lectures")


def find_relevant_lectures(query: str, top_n: int = 5) -> list[tuple[str, float]]:
    """
    Find the most relevant lectures for a query based on keyword-signature
    cosine similarity.

    Args:
        query:  The student's question.
        top_n:  Number of top lectures to return.

    Returns:
        List of (lecture_name, similarity_score) tuples, sorted by
        relevance (highest first).
    """
    if not _lecture_embeddings or _embedder is None:
        return []

    query_emb = _embedder.encode(query)

    scores: list[tuple[str, float]] = []
    for lecture_name, lecture_emb in _lecture_embeddings.items():
        # Cosine similarity
        dot = np.dot(query_emb, lecture_emb)
        norm = np.linalg.norm(query_emb) * np.linalg.norm(lecture_emb)
        sim = float(dot / norm) if norm > 0 else 0.0
        scores.append((lecture_name, sim))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_n]


def get_lecture_keywords(lecture_name: str) -> list[str]:
    """Return the TF-IDF keywords for a specific lecture."""
    return list(_lecture_keywords.get(lecture_name, []))


def get_all_lecture_names() -> list[str]:
    """Return a sorted list of all loaded lecture names."""
    return sorted(_lecture_keywords.keys())


def get_lecture_topics_summary() -> str:
    """
    Return a human-readable summary of all lectures and their top keywords.
    Useful for providing context to the supervisor agent.
    """
    if not _lecture_keywords:
        return "No lecture data loaded."

    lines: list[str] = []
    for name in sorted(_lecture_keywords.keys()):
        kw = _lecture_keywords[name][:10]  # Top 10 keywords
        lines.append(f"  {name}: {', '.join(kw)}")
    return "\n".join(lines)
