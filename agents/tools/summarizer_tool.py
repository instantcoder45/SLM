"""
Chapter summarization tool.

Loads all text chunks for a requested chapter from the FAISS database,
joins them, and sends them to the LLM to produce a concise summary.
Caches results so each chapter is only summarized once per session.
"""

from langchain_core.tools import tool

# ---------------------------------------------------------------------------
# Module-level state (populated by init_summarizer_tool)
# ---------------------------------------------------------------------------
_config = None
_summary_cache: dict[int, str] = {}   # chapter_number -> summary text

# Chapter titles for richer context
_CHAPTER_TITLES: dict[int, str] = {
    1: "Fundamentals of Quantitative Design and Analysis",
    2: "Memory Hierarchy Design",
    3: "Instruction-Level Parallelism and Its Exploitation",
    4: "Data-Level Parallelism in Vector, SIMD, and GPU Architectures",
    5: "Thread-Level Parallelism",
    6: "Warehouse-Scale Computers to Exploit Request-Level and Data-Level Parallelism",
}


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
def summarize_chapter(chapter_number: int) -> str:
    """Generate a comprehensive summary of a textbook chapter.

    Use this tool when the user asks for an overview, summary, or key
    points of a specific chapter from the 'Computer Architecture:
    A Quantitative Approach' textbook.

    The textbook has 6 chapters:
      1. Fundamentals of Quantitative Design and Analysis
      2. Memory Hierarchy Design
      3. Instruction-Level Parallelism and Its Exploitation
      4. Data-Level Parallelism in Vector, SIMD, and GPU Architectures
      5. Thread-Level Parallelism
      6. Warehouse-Scale Computers to Exploit Request-Level and Data-Level Parallelism

    Args:
        chapter_number: An integer from 1 to 6 indicating the chapter.

    Returns:
        A structured summary of the chapter, or an error message.
    """
    try:
        # Validate input
        if not isinstance(chapter_number, int) or chapter_number < 1 or chapter_number > 6:
            return (
                f"Invalid chapter number: {chapter_number}. "
                "Please provide an integer between 1 and 6."
            )

        # Return cached summary if available
        if chapter_number in _summary_cache:
            return (
                f"[Cached] Summary of Chapter {chapter_number}: "
                f"{_CHAPTER_TITLES.get(chapter_number, '')}\n\n"
                f"{_summary_cache[chapter_number]}"
            )

        # Lazy import to avoid circular dependency
        from agents.tools.rag_tool import get_chapter_texts
        from agents.llm import generate_with_chat_template

        # Load all text chunks for this chapter
        texts = get_chapter_texts(chapter_number)
        if not texts:
            return (
                f"No text data found for Chapter {chapter_number}. "
                "Make sure the chapter database is properly loaded."
            )

        # Combine chunks (truncate if too long to fit in context window)
        combined = "\n\n".join(texts)
        # Rough token estimate: ~4 chars/token, keep under ~3000 tokens for the context
        max_chars = 12000
        if len(combined) > max_chars:
            combined = combined[:max_chars] + "\n\n[... text truncated for length ...]"

        chapter_title = _CHAPTER_TITLES.get(chapter_number, f"Chapter {chapter_number}")

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a Computer Architecture professor. Provide a clear, "
                    "well-structured summary of the textbook chapter content provided below. "
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
                    f"Please summarize Chapter {chapter_number}: '{chapter_title}' "
                    f"from 'Computer Architecture: A Quantitative Approach'.\n\n"
                    f"Chapter content:\n{combined}"
                ),
            },
        ]

        summary = generate_with_chat_template(messages, max_new_tokens=600)

        # Cache the result
        _summary_cache[chapter_number] = summary

        return (
            f"Summary of Chapter {chapter_number}: {chapter_title}\n\n"
            f"{summary}"
        )

    except Exception as e:
        return f"Error summarizing chapter: {type(e).__name__}: {e}"
