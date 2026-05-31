"""
RAG retrieval tool for the Computer Architecture lecture slides.

Searches across all lecture FAISS vector databases to find the most
relevant text chunks for a given query.  Uses SentenceTransformer embeddings
and L2 distance for similarity ranking.

Supports lecture-focused retrieval via the lecture_registry module:
when enabled, only the most relevant lectures are searched instead of
brute-forcing all indices.
"""

import os
import re
import pickle
import numpy as np
from typing import Optional

from langchain_core.tools import tool

# ---------------------------------------------------------------------------
# Module-level state (populated by init_rag_tool)
# ---------------------------------------------------------------------------
_chapters: dict = {}          # {"Lecture-1": {"index": faiss.Index, "texts": [...], "meta": {...}}, ...}
_embedder = None              # SentenceTransformer instance
_top_k: int = 5               # Default number of results to return


def init_rag_tool(config) -> None:
    """
    Initialize the RAG tool by loading all lecture FAISS indices and texts.

    Must be called once at startup before `search_textbook` is used.

    Args:
        config: An agents.config.Config instance with `chapters_db_path`,
                `embed_model`, and `top_k` attributes.
    """
    global _chapters, _embedder, _top_k
    import faiss
    from sentence_transformers import SentenceTransformer

    _top_k = config.top_k

    # Load the embedding model
    print(f"📚 RAG Tool: Loading embedder '{config.embed_model}' ...")
    _embedder = SentenceTransformer(config.embed_model)

    # Load each lecture's FAISS index + texts
    chapters_root = config.chapters_db_path
    loaded = 0

    for entry in sorted(os.listdir(chapters_root)):
        chapter_path = os.path.join(chapters_root, entry)
        if not os.path.isdir(chapter_path):
            continue

        index_path = os.path.join(chapter_path, "index.faiss")
        texts_path = os.path.join(chapter_path, "texts.pkl")
        meta_path = os.path.join(chapter_path, "meta.pkl")

        if not all(os.path.exists(p) for p in [index_path, texts_path, meta_path]):
            print(f"   ⚠️  Skipping {entry}: missing files")
            continue

        index = faiss.read_index(index_path)
        with open(texts_path, "rb") as f:
            texts = pickle.load(f)
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        _chapters[entry] = {
            "index": index,
            "texts": texts,
            "meta": meta,
        }
        loaded += 1

    print(f"✅ RAG Tool: Loaded {loaded} lecture indices from {chapters_root}")


def _retrieve(
    query: str,
    top_k: Optional[int] = None,
    target_lectures: Optional[list[str]] = None,
) -> list[dict]:
    """
    Internal retrieval function.  Searches FAISS indices and returns the
    top_k most relevant chunks sorted by L2 distance.

    Args:
        query:            The search query string.
        top_k:            Number of results to return (default: config.top_k).
        target_lectures:  If provided, restrict search to only these lecture
                          folder names.  When None, searches all lectures.

    Returns:
        List of dicts with keys: chapter, text, distance
    """
    if not _chapters or _embedder is None:
        raise RuntimeError("RAG tool not initialized. Call init_rag_tool(config) first.")

    k = top_k or _top_k
    query_embedding = _embedder.encode([query]).astype("float32")

    # Decide which lectures to search
    if target_lectures:
        search_chapters = {
            name: data for name, data in _chapters.items()
            if name in target_lectures
        }
        # If filtering yielded nothing (bad names), fall back to all
        if not search_chapters:
            search_chapters = _chapters
    else:
        search_chapters = _chapters

    all_results: list[dict] = []

    for chapter_name, chapter_data in search_chapters.items():
        index = chapter_data["index"]
        texts = chapter_data["texts"]

        search_k = min(k, len(texts))
        distances, indices = index.search(query_embedding, search_k)

        for dist, idx in zip(distances[0], indices[0]):
            if 0 <= idx < len(texts):
                all_results.append({
                    "chapter": chapter_name,
                    "text": texts[idx],
                    "distance": float(dist),
                })

    # Sort by L2 distance (lower = more similar)
    all_results.sort(key=lambda x: x["distance"])
    return all_results[:k]


def get_chapter_texts(chapter_identifier) -> list[str]:
    """
    Return all text chunks for a given chapter/lecture.

    Supports multiple identifier formats:
      - Exact folder name: "Lecture-12", "L-14", "RARS"
      - Numeric: 12 → tries "Lecture-12", "L-12", "Lec12"
      - String number: "12" → same fuzzy matching as numeric

    Args:
        chapter_identifier: Lecture name string, or integer/string number.

    Returns:
        List of text strings for that lecture, or empty list if not found.
    """
    # Try exact match first
    identifier = str(chapter_identifier)
    if identifier in _chapters:
        return list(_chapters[identifier]["texts"])

    # Try common naming patterns
    patterns_to_try = [
        identifier,
        f"Lecture-{identifier}",
        f"L-{identifier}",
        f"Lec{identifier}",
        f"chapter{identifier}",
        f"Chapter_{identifier}",
    ]

    for pattern in patterns_to_try:
        if pattern in _chapters:
            return list(_chapters[pattern]["texts"])

    # Fuzzy: try case-insensitive substring matching
    id_lower = identifier.lower()
    for name in _chapters:
        if id_lower in name.lower() or name.lower() in id_lower:
            return list(_chapters[name]["texts"])

    return []


def get_matching_lecture_name(identifier) -> Optional[str]:
    """
    Resolve a lecture identifier to the actual folder name in _chapters.

    Returns the matched name or None if not found.
    """
    identifier = str(identifier)
    if identifier in _chapters:
        return identifier

    patterns_to_try = [
        identifier,
        f"Lecture-{identifier}",
        f"L-{identifier}",
        f"Lec{identifier}",
        f"chapter{identifier}",
        f"Chapter_{identifier}",
    ]
    for pattern in patterns_to_try:
        if pattern in _chapters:
            return pattern

    id_lower = identifier.lower()
    for name in _chapters:
        if id_lower in name.lower() or name.lower() in id_lower:
            return name

    return None


def get_all_loaded_lectures() -> list[str]:
    """Return sorted list of all loaded lecture/chapter names."""
    return sorted(_chapters.keys())


@tool
def search_textbook(query: str) -> str:
    """Search the Computer Architecture lecture slides for information relevant to a query.

    Use this tool when you need to find specific information from the
    lecture slides. The tool searches across relevant lectures using
    semantic similarity and returns the most relevant passages with
    source citations.

    The tool automatically identifies which lectures are most likely to
    contain the answer using keyword-signature matching, then performs
    a focused FAISS vector search within those lectures.

    Args:
        query: A natural language question or topic to search for.
               Example: 'What are the five stages of a MIPS pipeline?'

    Returns:
        Formatted string of the top matching passages, each prefixed with
        its source lecture. Returns an error message if retrieval fails.
    """
    try:
        # Use lecture-focused retrieval if the registry is available
        target_lectures = None
        try:
            from agents.tools.lecture_registry import find_relevant_lectures
            relevant = find_relevant_lectures(query, top_n=5)
            if relevant:
                target_lectures = [name for name, _score in relevant]
                print(f"   🎯 Focused search on: {target_lectures}")
        except Exception:
            pass  # Registry not available — search all lectures

        results = _retrieve(query, target_lectures=target_lectures)

        if not results:
            return "No relevant passages found in the lecture slides for this query."

        # Format results with source citations
        formatted_parts: list[str] = []
        for i, r in enumerate(results, 1):
            lecture_label = r["chapter"]
            score = r["distance"]
            text = r["text"].strip()
            formatted_parts.append(
                f"[Source: {lecture_label}] (relevance score: {score:.2f})\n{text}"
            )

        return "\n\n---\n\n".join(formatted_parts)

    except RuntimeError as e:
        return f"RAG tool error: {e}"
    except Exception as e:
        return f"Error searching lecture slides: {type(e).__name__}: {e}"
