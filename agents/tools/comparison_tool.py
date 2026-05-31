"""
Concept comparison tool.

Retrieves textbook context for two Computer Architecture concepts via the
RAG tool, then uses the LLM to generate a structured side-by-side comparison.
"""

from langchain_core.tools import tool

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------
_config = None


def init_comparison_tool(config) -> None:
    """
    Initialize the comparison tool with the system configuration.

    Args:
        config: An agents.config.Config instance.
    """
    global _config
    _config = config
    print("✅ Comparison tool initialized")


@tool
def compare_concepts(concept_a: str, concept_b: str) -> str:
    """Compare two Computer Architecture concepts side by side.

    Use this tool when the user asks to compare, contrast, or differentiate
    two technical concepts (e.g., 'RISC vs CISC', 'write-back vs write-through',
    'SIMD vs MIMD').

    The tool retrieves relevant textbook context for both concepts and then
    generates a structured comparison covering key dimensions such as
    definition, advantages, disadvantages, use cases, and performance
    characteristics.

    Args:
        concept_a: The first concept to compare (e.g., 'direct-mapped cache').
        concept_b: The second concept to compare (e.g., 'set-associative cache').

    Returns:
        A structured comparison table/analysis, or an error message.
    """
    try:
        # Lazy imports to avoid circular dependencies
        from agents.tools.rag_tool import _retrieve
        from agents.llm import generate_with_chat_template

        # Retrieve context for both concepts
        results_a = _retrieve(concept_a, top_k=3)
        results_b = _retrieve(concept_b, top_k=3)

        # Build context strings
        context_a_parts = []
        for r in results_a:
            ch_label = r["chapter"].replace("chapter", "Chapter ")
            context_a_parts.append(f"[{ch_label}] {r['text'].strip()}")
        context_a = "\n\n".join(context_a_parts) if context_a_parts else "No textbook context found."

        context_b_parts = []
        for r in results_b:
            ch_label = r["chapter"].replace("chapter", "Chapter ")
            context_b_parts.append(f"[{ch_label}] {r['text'].strip()}")
        context_b = "\n\n".join(context_b_parts) if context_b_parts else "No textbook context found."

        # Truncate contexts to stay within token limits
        max_ctx_chars = 4000
        if len(context_a) > max_ctx_chars:
            context_a = context_a[:max_ctx_chars] + "\n[...truncated...]"
        if len(context_b) > max_ctx_chars:
            context_b = context_b[:max_ctx_chars] + "\n[...truncated...]"

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a Computer Architecture professor. Compare the two concepts "
                    "provided by the student using a structured format:\n\n"
                    "1. **Brief definition** of each concept\n"
                    "2. **Comparison table** with rows for key dimensions such as:\n"
                    "   - Design philosophy / approach\n"
                    "   - Advantages\n"
                    "   - Disadvantages\n"
                    "   - Performance characteristics\n"
                    "   - Use cases / examples\n"
                    "3. **Key takeaway** — when to prefer one over the other\n\n"
                    "Use the provided textbook context to ground your comparison. "
                    "Use markdown formatting for the table."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Compare **{concept_a}** vs **{concept_b}**.\n\n"
                    f"--- Context for {concept_a} ---\n{context_a}\n\n"
                    f"--- Context for {concept_b} ---\n{context_b}"
                ),
            },
        ]

        comparison = generate_with_chat_template(messages, max_new_tokens=700)

        return (
            f"Comparison: {concept_a} vs {concept_b}\n"
            f"{'=' * 50}\n\n"
            f"{comparison}"
        )

    except Exception as e:
        return f"Error comparing concepts: {type(e).__name__}: {e}"
