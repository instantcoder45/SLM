"""
Lecture summarization tool.

Loads all text chunks for a requested lecture from the FAISS database,
joins them, and sends them to the LLM to produce a concise summary.
Caches results so each lecture is only summarized once per session.
"""

from langchain_core.tools import tool

# ---------------------------------------------------------------------------
# Module-level state (populated by init_summarizer_tool)
# ---------------------------------------------------------------------------
_config = None
_summary_cache: dict[str, str] = {}   # lecture_identifier -> summary text


def init_summarizer_tool(config) -> None:
    """
    Initialize the summarizer tool with the system configuration.

    Args:
        config: An agents.config.Config instance.
    """
    global _config
    _config = config
    print("✅ Summarizer tool initialized")


@tool
def summarize_chapter(chapter_identifier: str) -> str:
    """Generate a comprehensive summary of a lecture or chapter.

    Use this tool when the user asks for an overview, summary, or key
    points of a specific lecture from the Computer Architecture course.

    Accepts flexible identifiers:
      - Lecture name: "Lecture-12", "L-14", "RARS", "Lec21"
      - Lecture number: "12", "14"

    Args:
        chapter_identifier: A string identifying the lecture (name or number).

    Returns:
        A structured summary of the lecture, or an error message.
    """
    try:
        # Lazy imports to avoid circular dependency
        from agents.tools.rag_tool import get_chapter_texts, get_matching_lecture_name
        from agents.llm import generate_with_chat_template

        # Resolve the identifier to an actual lecture name
        identifier = str(chapter_identifier).strip()
        matched_name = get_matching_lecture_name(identifier)

        if matched_name is None:
            # Try to provide helpful suggestions
            from agents.tools.rag_tool import get_all_loaded_lectures
            available = get_all_loaded_lectures()
            suggestion = ", ".join(available[:10])
            return (
                f"Could not find a lecture matching '{identifier}'. "
                f"Available lectures include: {suggestion}... "
                f"Please specify an exact lecture name or number."
            )

        # Return cached summary if available
        if matched_name in _summary_cache:
            return (
                f"[Cached] Summary of {matched_name}\n\n"
                f"{_summary_cache[matched_name]}"
            )

        # Load all text chunks for this lecture
        texts = get_chapter_texts(matched_name)
        if not texts:
            return (
                f"No text data found for '{matched_name}'. "
                "Make sure the lecture database is properly loaded."
            )

        # Combine chunks (truncate if too long to fit in context window)
        combined = "\n\n".join(texts)
        # Rough token estimate: ~4 chars/token, keep under ~3000 tokens for the context
        max_chars = 12000
        if len(combined) > max_chars:
            combined = combined[:max_chars] + "\n\n[... text truncated for length ...]"

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a Computer Architecture professor. Provide a clear, "
                    "well-structured summary of the lecture content provided below. "
                    "Organise your summary with:\n"
                    "1. A one-paragraph overview\n"
                    "2. Key topics covered (bulleted list)\n"
                    "3. Important concepts and takeaways\n"
                    "Be concise but thorough. Use technical terminology accurately."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Please summarize the lecture '{matched_name}' "
                    f"from the Computer Architecture course.\n\n"
                    f"Lecture content:\n{combined}"
                ),
            },
        ]

        summary = generate_with_chat_template(messages, max_new_tokens=600)

        # Cache the result
        _summary_cache[matched_name] = summary

        return (
            f"Summary of {matched_name}\n\n"
            f"{summary}"
        )

    except Exception as e:
        return f"Error summarizing lecture: {type(e).__name__}: {e}"
