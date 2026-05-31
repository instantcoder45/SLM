"""
Knowledge specialist agent.

Handles three sub-tasks via keyword-based internal routing:
    1. lookup_definition  – "define …", "what is …", "meaning of …"
    2. summarize_chapter  – "summarize lecture N", "overview of lecture N"
    3. compare_concepts   – "compare X vs Y", "difference between X and Y"

After calling the appropriate tool the LLM formats the raw output into a
clear, student-friendly response.
"""

from __future__ import annotations

import re
import traceback
from typing import Any

from agents.state import AgentState
from agents.llm import generate_with_chat_template

# ---------------------------------------------------------------------------
# Lazy tool imports
# ---------------------------------------------------------------------------

def _get_lookup_definition():
    from agents.tools import lookup_definition
    return lookup_definition

def _get_summarize_chapter():
    from agents.tools import summarize_chapter
    return summarize_chapter

def _get_compare_concepts():
    from agents.tools import compare_concepts
    return compare_concepts


# ---------------------------------------------------------------------------
# Sub-routing helpers
# ---------------------------------------------------------------------------

_DEFINITION_KEYWORDS = [
    "define", "what is", "what are", "meaning of", "meaning",
    "definition of", "definition", "explain the term",
]

_SUMMARY_KEYWORDS = [
    "summarize", "summary", "summarise", "overview of chapter",
    "overview chapter", "chapter summary", "recap chapter",
    "overview of lecture", "overview lecture", "lecture summary",
    "recap lecture", "summarize lecture", "summarise lecture",
]

_COMPARE_KEYWORDS = [
    "compare", "comparison", "difference between", "differences between",
    "vs", "versus", " vs ", "contrast",
]


def _detect_sub_task(query: str) -> str:
    """Determine which knowledge sub-task to invoke."""
    q_lower = query.lower()

    # Check summaries first (more specific keyword set)
    for kw in _SUMMARY_KEYWORDS:
        if kw in q_lower:
            return "summarize"

    # Check comparisons
    for kw in _COMPARE_KEYWORDS:
        if kw in q_lower:
            return "compare"

    # Check definitions (broadest – default within knowledge)
    for kw in _DEFINITION_KEYWORDS:
        if kw in q_lower:
            return "define"

    # Default sub-task: treat as definition lookup
    return "define"


def _extract_lecture_identifier(query: str) -> str | None:
    """
    Try to extract a lecture/chapter identifier from the query string.

    Supports formats like:
      - "Lecture 12", "Lecture-12"
      - "L-14", "L 14"
      - "Lec21", "Lec 21"
      - "chapter 3", "chapter3"
      - Just a number if preceded by summary keywords
    """
    # Try "Lecture-N" or "Lecture N"
    match = re.search(r"lecture[- ]?(\d+)", query, re.IGNORECASE)
    if match:
        return f"Lecture-{match.group(1)}"

    # Try "L-N" or "L N"
    match = re.search(r"\bL[- ]?(\d+)", query)
    if match:
        return f"L-{match.group(1)}"

    # Try "Lec N" or "LecN"
    match = re.search(r"\blec[- ]?(\d+)", query, re.IGNORECASE)
    if match:
        return f"Lec{match.group(1)}"

    # Try "chapter N"
    match = re.search(r"chapter\s*(\d+)", query, re.IGNORECASE)
    if match:
        return match.group(1)

    # Try "RARS" explicitly
    if re.search(r"\brars\b", query, re.IGNORECASE):
        return "RARS"

    # Try bare number near summary keywords
    match = re.search(r"\b(\d+)\b", query)
    if match:
        return match.group(1)

    return None


def _extract_term(query: str) -> str:
    """Extract the term to define from the query."""
    q_lower = query.lower()
    for kw in _DEFINITION_KEYWORDS:
        if kw in q_lower:
            # Take everything after the keyword
            idx = q_lower.index(kw) + len(kw)
            term = query[idx:].strip().strip("?.,!\"'")
            if term:
                return term
    # Fallback: use the whole query
    return query.strip().strip("?.,!\"'")


def _extract_concepts(query: str) -> tuple[str, str]:
    """Extract two concepts to compare from the query."""
    q_lower = query.lower()

    # Try "X vs Y" or "X versus Y"
    for sep in [" vs ", " versus ", " vs. "]:
        if sep in q_lower:
            parts = query.split(sep if sep in query else sep.title(), 1)
            if len(parts) == 2:
                # Strip leading keywords from the first part
                a = parts[0].strip()
                for kw in _COMPARE_KEYWORDS:
                    if a.lower().startswith(kw):
                        a = a[len(kw):].strip()
                return a.strip("?.,!\"' "), parts[1].strip("?.,!\"' ")

    # Try "difference(s) between X and Y"
    match = re.search(
        r"(?:difference|differences|compare|contrast)\s+(?:between\s+)?(.+?)\s+and\s+(.+)",
        query,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).strip("?.,!\"' "), match.group(2).strip("?.,!\"' ")

    # Fallback: split on "and"
    if " and " in q_lower:
        parts = query.split(" and ", 1)
        a = parts[0].strip()
        for kw in _COMPARE_KEYWORDS:
            if a.lower().startswith(kw):
                a = a[len(kw):].strip()
        return a.strip("?.,!\"' "), parts[1].strip("?.,!\"' ")

    return query, ""


# ---------------------------------------------------------------------------
# Formatting prompt
# ---------------------------------------------------------------------------

_FORMAT_SYSTEM = """\
You are a Computer Architecture teaching assistant.  The student asked a
knowledge question and a tool has returned raw information.  Your job is
to present the information in a clear, well-structured, student-friendly
format.  Use bullet points, numbered lists, or short paragraphs as appropriate.
"""


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def knowledge_agent_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node – Knowledge specialist.

    Internally routes to lookup_definition, summarize_chapter, or
    compare_concepts based on keyword detection, then formats the result.
    """
    query: str = state.get("current_query", "")
    conversation: str = state.get("conversation_context", "No previous conversation.")
    tool_log: list[str] = list(state.get("tool_calls_log", []))
    sources: list[dict] = list(state.get("sources", []))

    try:
        sub_task = _detect_sub_task(query)
        raw_result = ""

        # ---- Dispatch to the appropriate tool ----

        if sub_task == "summarize":
            lecture_id = _extract_lecture_identifier(query)
            if lecture_id is None:
                raw_result = "I couldn't determine which lecture to summarize. Please specify a lecture name or number (e.g. 'summarize Lecture 12' or 'summarize L-14')."
            else:
                summarize = _get_summarize_chapter()
                raw_result = summarize.invoke(lecture_id)
                tool_log.append(f"summarize_chapter(identifier={lecture_id!r})")
                sources.append({"chapter": lecture_id, "type": "summary"})

        elif sub_task == "compare":
            concept_a, concept_b = _extract_concepts(query)
            if not concept_b:
                raw_result = "I couldn't identify two concepts to compare. Please phrase like 'compare X vs Y'."
            else:
                compare = _get_compare_concepts()
                raw_result = compare.invoke({"concept_a": concept_a, "concept_b": concept_b})
                tool_log.append(f"compare_concepts(a={concept_a!r}, b={concept_b!r})")
                sources.append({"type": "comparison", "concepts": [concept_a, concept_b]})

        else:  # "define"
            term = _extract_term(query)
            lookup = _get_lookup_definition()
            raw_result = lookup.invoke(term)
            tool_log.append(f"lookup_definition(term={term!r})")
            sources.append({"type": "definition", "term": term})

        # ---- Format with LLM ----
        format_messages = [
            {"role": "system", "content": _FORMAT_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Conversation context:\n{conversation}\n\n"
                    f"Student question: {query}\n\n"
                    f"Tool output ({sub_task}):\n{raw_result}\n\n"
                    "Format this into a clear answer for the student."
                ),
            },
        ]
        answer = generate_with_chat_template(format_messages, max_new_tokens=1024)

        print(f"📖 Knowledge agent ({sub_task}) produced answer ({len(answer)} chars)")

        return {
            "agent_output": answer,
            "tool_calls_log": tool_log,
            "sources": sources,
        }

    except Exception as exc:
        error_msg = f"Knowledge agent error: {exc}\n{traceback.format_exc()}"
        print(f"❌ {error_msg}")
        return {
            "agent_output": (
                "I'm sorry, I encountered an error while looking up that information. "
                "Please try rephrasing your question."
            ),
            "tool_calls_log": tool_log,
            "sources": sources,
            "error": error_msg,
        }
