"""
Supervisor / Router agent.

Uses a hybrid routing strategy:
  1. Fast regex check for greetings / chitchat  (instant, no LLM)
  2. LLM-based classification for all other queries
  3. Keyword-based fallback if the LLM call fails or returns junk

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


# ───────────────────────────────────────────────────────────────
# LLM-based routing
# ───────────────────────────────────────────────────────────────

_VALID_AGENTS = {"RAG_AGENT", "MATH_AGENT", "KNOWLEDGE_AGENT", "CODE_AGENT", "DIRECT"}
_AGENT_EXTRACT = re.compile(
    r"(RAG_AGENT|MATH_AGENT|KNOWLEDGE_AGENT|CODE_AGENT|DIRECT)"
)

_ROUTER_SYSTEM_PROMPT = """\
You are a query router. Classify the user's query into exactly ONE of the following categories. Reply with ONLY the category name and nothing else.

RAG_AGENT – Conceptual questions about computer architecture topics, theory, how things work, explaining concepts.
  Examples: "How does pipelining work?", "Explain virtual memory", "Why is cache important?"

MATH_AGENT – Numerical calculations, performance metrics, CPI, speedup, Amdahl's Law, formulas.
  Examples: "Calculate the CPI given these values", "What is the speedup with 4 processors?", "Compute the execution time"

KNOWLEDGE_AGENT – Definitions ("define X", "what is X"), chapter/lecture summaries, concept comparisons ("compare X vs Y").
  Examples: "Define cache coherence", "Summarize chapter 5", "Compare RISC vs CISC"

CODE_AGENT – Assembly language code: writing, tracing, or explaining MIPS/ARM/x86 instructions.
  Examples: "Write a MIPS program to add two numbers", "Trace this assembly code", "Explain the BEQ instruction"

DIRECT – Greetings, chitchat, thank-yous, non-technical small talk.
  Examples: "Hello", "Thanks!", "How are you?"

Respond with ONLY the category name."""


def _llm_route(query: str, conversation_context: str) -> str | None:
    """Classify the query using the LLM. Returns an agent name or None."""
    from agents.llm import generate_with_chat_template

    messages = [
        {"role": "system", "content": _ROUTER_SYSTEM_PROMPT},
    ]
    if conversation_context:
        messages.append(
            {"role": "user", "content": f"Conversation context:\n{conversation_context}"}
        )
    messages.append({"role": "user", "content": query})

    response = generate_with_chat_template(messages, max_new_tokens=10)

    match = _AGENT_EXTRACT.search(response)
    if match:
        return match.group(1)
    return None


# ───────────────────────────────────────────────────────────────
# LangGraph node
# ───────────────────────────────────────────────────────────────

def supervisor_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node – Supervisor / Router.

    Hybrid strategy:
      1. Regex for instant chitchat detection
      2. LLM-based classification
      3. Keyword fallback if LLM fails
    """
    query: str = state.get("current_query", "")

    if not query.strip():
        return {
            "selected_agent": "DIRECT",
            "error": "Empty query received by supervisor.",
        }

    # ── Step 1: instant chitchat check (no LLM needed) ──
    if _DIRECT_PATTERNS.match(query.strip()) or len(query.strip()) < 4:
        print("🔀 Supervisor routed to: DIRECT  (regex)")
        return {"selected_agent": "DIRECT"}

    # ── Step 2: LLM-based classification ──
    conversation_context = state.get("conversation_context", "")
    try:
        llm_choice = _llm_route(query, conversation_context)
        if llm_choice and llm_choice in _VALID_AGENTS:
            print(f"🔀 Supervisor routed to: {llm_choice}  (LLM)")
            return {"selected_agent": llm_choice}
    except Exception as exc:  # noqa: BLE001
        print(f"⚠️  LLM routing failed ({exc}), falling back to keywords.")

    # ── Step 3: keyword fallback ──
    selected = _keyword_route(query)
    print(f"🔀 Supervisor routed to: {selected}  (keyword fallback)")
    return {"selected_agent": selected}
