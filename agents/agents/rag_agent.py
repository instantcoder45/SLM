"""
RAG (Retrieval-Augmented Generation) specialist agent.

Retrieves relevant textbook passages via FAISS search, then uses the LLM
to synthesise a grounded answer.  Falls back to web search when the
textbook context is insufficient.
"""

from __future__ import annotations

import traceback
from typing import Any

from agents.state import AgentState
from agents.llm import generate_with_chat_template

# ---------------------------------------------------------------------------
# Lazy tool imports – resolved at call-time to avoid circular import issues
# and to allow the graph builder to initialise tools first.
# ---------------------------------------------------------------------------

def _get_search_textbook():
    from agents.tools import search_textbook
    return search_textbook

def _get_web_search():
    from agents.tools import web_search
    return web_search


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a Computer Architecture teaching assistant.  Answer the student's
question using ONLY the provided textbook context.  Be clear and concise.

Rules:
- Get straight to the answer. Do not start with greetings or "Here is...".
- Use bullet points and short paragraphs.
- Cite the source chapter when possible (e.g. "According to Chapter 3…").
- If the context does not contain enough information, say so honestly.
- Use examples or analogies to aid understanding when appropriate.
"""


def _build_answer_prompt(query: str, context: str, conversation: str) -> list[dict]:
    """Compose chat messages for the answer-generation step."""
    user_content = (
        f"Conversation so far:\n{conversation}\n\n"
        f"Retrieved textbook context:\n{context}\n\n"
        f"Student question: {query}\n\n"
        "Provide a clear, well-structured answer."
    )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def rag_agent_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node – RAG specialist.

    1. Searches the textbook FAISS index for relevant chunks.
    2. If retrieval is poor (short / empty), tries web_search as fallback.
    3. Uses the LLM to generate a contextual answer.
    4. Records sources and tool calls in state.
    """
    query: str = state.get("current_query", "")
    conversation: str = state.get("conversation_context", "No previous conversation.")
    tool_log: list[str] = list(state.get("tool_calls_log", []))
    sources: list[dict] = list(state.get("sources", []))

    try:
        # ---- Step 1: Textbook retrieval ----
        search_textbook = _get_search_textbook()
        retrieval_result = search_textbook.invoke(query)
        tool_log.append(f"search_textbook(query={query!r})")

        # ---- Step 2: Evaluate quality & optional fallback ----
        context = retrieval_result
        source_type = "textbook"

        if len(retrieval_result.strip()) < 80:
            # Retrieval looks thin – try web search as supplement
            try:
                web_search = _get_web_search()
                web_result = web_search.invoke(query)
                tool_log.append(f"web_search(query={query!r})  [fallback]")
                if web_result and len(web_result.strip()) > 40:
                    context = (
                        f"--- Textbook context ---\n{retrieval_result}\n\n"
                        f"--- Web context (supplementary) ---\n{web_result}"
                    )
                    source_type = "textbook+web"
            except Exception:
                pass  # web search failing is non-critical

        # ---- Step 3: Build source citations ----
        # Parse chapter references from the context string
        import re
        chapter_refs = re.findall(r"\[Source:\s*(chapter\d+)\]", context, re.IGNORECASE)
        for ch in chapter_refs:
            sources.append({"chapter": ch, "type": source_type})
        if not chapter_refs:
            sources.append({"type": source_type, "note": "Chapters could not be identified from context."})

        # ---- Step 4: Generate answer ----
        messages = _build_answer_prompt(query, context, conversation)
        answer = generate_with_chat_template(messages, max_new_tokens=1024)

        print(f"📚 RAG agent produced answer ({len(answer)} chars, {len(chapter_refs)} sources)")

        return {
            "agent_output": answer,
            "tool_calls_log": tool_log,
            "sources": sources,
        }

    except Exception as exc:
        error_msg = f"RAG agent error: {exc}\n{traceback.format_exc()}"
        print(f"❌ {error_msg}")
        # Fallback: try answering from LLM knowledge alone
        try:
            fallback_messages = [
                {"role": "system", "content": "You are a Computer Architecture teaching assistant. Answer the student's question concisely using your knowledge. Get straight to the answer."},
                {"role": "user", "content": state.get("current_query", "")},
            ]
            fallback_answer = generate_with_chat_template(fallback_messages, max_new_tokens=1024)
            return {
                "agent_output": fallback_answer,
                "tool_calls_log": tool_log + ["FALLBACK: LLM-only (RAG search failed)"],
                "sources": sources,
            }
        except Exception:
            return {
                "agent_output": "I encountered an error while searching the textbook. Please try rephrasing your question.",
                "tool_calls_log": tool_log,
                "sources": sources,
                "error": error_msg,
            }
