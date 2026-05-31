"""
RAG retrieval tool for the Computer Architecture textbook.

Searches across all 6 chapter FAISS vector databases to find the most
relevant text chunks for a given query. Uses SentenceTransformer embeddings
and L2 distance for similarity ranking.
"""

import os
import pickle
import numpy as np
from typing import Optional

from langchain_core.tools import tool

# ---------------------------------------------------------------------------
# Module-level state (populated by init_rag_tool)
# ---------------------------------------------------------------------------
_chapters: dict = {}          # {"chapter1": {"index": faiss.Index, "texts": [...], "meta": {...}}, ...}
_embedder = None              # SentenceTransformer instance
_top_k: int = 5               # Default number of results to return


def init_rag_tool(config) -> None:
    """
    Initialize the RAG tool by loading all chapter FAISS indices and texts.

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

    # Load each chapter's FAISS index + texts
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

    print(f"✅ RAG Tool: Loaded {loaded} chapter indices from {chapters_root}")


def _retrieve(query: str, top_k: Optional[int] = None) -> list[dict]:
    """
    Internal retrieval function. Searches all loaded chapter FAISS indices
    and returns the top_k most relevant chunks sorted by L2 distance.

    Returns:
        List of dicts with keys: chapter, text, distance
    """
    if not _chapters or _embedder is None:
        raise RuntimeError("RAG tool not initialized. Call init_rag_tool(config) first.")

    k = top_k or _top_k
    query_embedding = _embedder.encode([query]).astype("float32")

    all_results: list[dict] = []

    for chapter_name, chapter_data in _chapters.items():
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


def get_chapter_texts(chapter_number: int) -> list[str]:
    """
    Return all text chunks for a given chapter number.
    Useful for the summarizer tool.

    Args:
        chapter_number: Integer 1-6.

    Returns:
        List of text strings for that chapter, or empty list if not found.
    """
    key = f"chapter{chapter_number}"
    if key in _chapters:
        return list(_chapters[key]["texts"])
    return []


@tool
def search_textbook(query: str) -> str:
    """Search the Computer Architecture textbook for information relevant to a query.

    Use this tool when you need to find specific information from the
    'Computer Architecture: A Quantitative Approach' textbook. The tool
    searches across all 6 chapters using semantic similarity and returns
    the most relevant passages with source citations.

    Args:
        query: A natural language question or topic to search for.
               Example: 'What are the five stages of a MIPS pipeline?'

    Returns:
        Formatted string of the top matching passages, each prefixed with
        its source chapter. Returns an error message if retrieval fails.
    """
    try:
        results = _retrieve(query)

        if not results:
            return "No relevant passages found in the textbook for this query."

        # Format results with source citations
        formatted_parts: list[str] = []
        for i, r in enumerate(results, 1):
            # Convert directory name like 'chapter3' -> 'Chapter 3'
            ch_label = r["chapter"].replace("chapter", "Chapter ")
            score = r["distance"]
            text = r["text"].strip()
            formatted_parts.append(
                f"[Source: {ch_label}] (relevance score: {score:.2f})\n{text}"
            )

        return "\n\n---\n\n".join(formatted_parts)

    except RuntimeError as e:
        return f"RAG tool error: {e}"
    except Exception as e:
        return f"Error searching textbook: {type(e).__name__}: {e}"
