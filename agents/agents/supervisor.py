"""
Supervisor / Router agent.

Uses fast keyword-based routing (no LLM call) to decide which
specialist agent should handle the request. Falls back to RAG_AGENT
for ambiguous queries.

Routing targets:
    RAG_AGENT         – Textbook concepts, theory, architecture topics
    MATH_AGENT        – Calculations, performance metrics, CPI, speedup
    KNOWLEDGE_AGENT   – Definitions, chapter summaries, concept comparisons
    CODE_AGENT        – Assembly code questions, tracing, writing code
    DIRECT            – Greetings, general chat, simple follow-ups
"""

from __future__ import annotations

import re
from typing import Any

from agents.state import AgentState


# ───────────────────────────────────────────────────────────────
# Keyword-based routing  (eliminates one LLM call ≈ 20-25 s)
# ───────────────────────────────────────────────────────────────

# Order matters: check more specific categories first.

_DIRECT_PATTERNS = re.compile(
    r"^(hi|hello|hey|thanks|thank you|bye|good morning|good evening|good night"
    r"|ok|okay|yes|no|sure|great|cool)\b",
    re.IGNORECASE,
)

_MATH_KEYWORDS = [
    "calculate", "compute", "cpi", "speedup", "amdahl",
    "throughput", "bandwidth", "clock cycle", "execution time",
    "how many cycles", "how many bits", "how many bytes",
    "performance", "miss rate", "hit rate", "miss penalty",
    "power consumption", "energy", "mips rating",
]
_MATH_PATTERN = re.compile(
    r"(?:" + "|".join(re.escape(k) for k in _MATH_KEYWORDS) + r")",
    re.IGNORECASE,
)
# Pure arithmetic: e.g. "2**10", "1024 / 8 + 3"
_PURE_MATH = re.compile(r"^[\d\s\+\-\*/\(\)\.\^%]+$")

_CODE_KEYWORDS = [
    "mips", "assembly", "asm", "instruction", "opcode",
    "trace", "register", "syscall", "beq", "bne", "addi",
    "write code", "write mips", "write assembly",
    "sll", "srl", "lw", "sw", "add ", "sub ", "lui", "ori",
    "arm", "x86", "machine code", "encoding",
    ".text", ".data", ".globl",
]
_CODE_PATTERN = re.compile(
    r"(?:" + "|".join(re.escape(k) for k in _CODE_KEYWORDS) + r")",
    re.IGNORECASE,
)

_KNOWLEDGE_KEYWORDS = [
    "define ", "definition", "what is a ", "what is the ",
    "what are ", "meaning of", "summarize", "summary",
    "overview of chapter", "compare", "vs", "versus",
    "difference between", "differences between",
]
_KNOWLEDGE_PATTERN = re.compile(
    r"(?:" + "|".join(re.escape(k) for k in _KNOWLEDGE_KEYWORDS) + r")",
    re.IGNORECASE,
)


def _keyword_route(query: str) -> str:
    """Classify the query using keyword matching."""
    q = query.strip()

    # Very short greetings / chitchat
    if _DIRECT_PATTERNS.match(q) or len(q) < 4:
        return "DIRECT"

    # Pure arithmetic expression
    if _PURE_MATH.fullmatch(q):
        return "MATH_AGENT"

    # Code / assembly
    if _CODE_PATTERN.search(q):
        return "CODE_AGENT"

    # Math / performance
    if _MATH_PATTERN.search(q):
        return "MATH_AGENT"

    # Definitions / summaries / comparisons
    if _KNOWLEDGE_PATTERN.search(q):
        return "KNOWLEDGE_AGENT"

    # Default: use RAG for textbook concept questions
    return "RAG_AGENT"


def supervisor_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node – Supervisor / Router.

    Uses keyword-based routing for speed (no LLM call).
    """
    query: str = state.get("current_query", "")

    if not query.strip():
        return {
            "selected_agent": "DIRECT",
            "error": "Empty query received by supervisor.",
        }

    selected = _keyword_route(query)

    print(f"🔀 Supervisor routed to: {selected}")

    return {"selected_agent": selected}
