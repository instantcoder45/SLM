"""
Code specialist agent.

Handles assembly-language questions:
    - Tracing assembly code step-by-step (MIPS, ARM, x86)
    - Writing assembly code from natural-language descriptions
    - Explaining instruction encoding / decoding
    - Looking up instruction-set references
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

def _get_trace_assembly():
    from agents.tools import trace_assembly_code
    return trace_assembly_code

def _get_search_textbook():
    from agents.tools import search_textbook
    return search_textbook

def _get_web_search():
    from agents.tools import web_search
    return web_search


# ---------------------------------------------------------------------------
# Heuristics for sub-task detection
# ---------------------------------------------------------------------------

_TRACE_KEYWORDS = [
    "trace", "step through", "execute", "walk through",
    "what happens when", "run this", "step-by-step",
    "what does this code", "output of",
]

_WRITE_KEYWORDS = [
    "write", "create", "generate", "code for", "implement",
    "assembly for", "asm for", "program to", "write a program",
    "write assembly", "write mips", "write arm",
]


def _contains_code_block(text: str) -> bool:
    """Check if the query appears to contain inline assembly code."""
    # Look for common assembly patterns
    asm_patterns = [
        r"\b(add|sub|mul|div|lw|sw|beq|bne|j|jal|jr|sll|srl|and|or|xor|nor|slt|addi|andi|ori|lui)\b",
        r"\b(mov|push|pop|call|ret|jmp|cmp|lea|inc|dec|nop)\b",
        r"\b(ADD|SUB|LDR|STR|MOV|B|BL|CMP)\b",
        r"\$[a-z0-9]+",    # MIPS registers like $t0, $s1
        r"\b[Rr]\d+\b",     # ARM registers like R0, R1
        r"%[a-z]+",          # x86 registers like %eax
    ]
    for pat in asm_patterns:
        if re.search(pat, text):
            return True
    return False


def _detect_code_sub_task(query: str) -> str:
    """Decide whether this is a trace, write, or explain task."""
    q_lower = query.lower()

    for kw in _TRACE_KEYWORDS:
        if kw in q_lower:
            return "trace"

    for kw in _WRITE_KEYWORDS:
        if kw in q_lower:
            return "write"

    # If the query contains what looks like assembly code, default to trace
    if _contains_code_block(query):
        return "trace"

    # Default: general code explanation
    return "explain"


def _extract_code_block(query: str) -> str:
    """Try to extract an assembly code block from the query."""
    # Look for fenced code blocks
    match = re.search(r"```[\w]*\n?(.*?)```", query, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Look for indented blocks (lines starting with spaces/tabs that look like asm)
    lines = query.split("\n")
    code_lines = []
    for line in lines:
        stripped = line.strip()
        if re.match(r"^(add|sub|mul|div|lw|sw|beq|bne|j |jal|jr|sll|srl|and|or|xor|nor|slt|addi|andi|ori|lui|mov|push|pop|call|ret|jmp|cmp|lea|nop|ADD|SUB|LDR|STR|MOV|B |BL|CMP)", stripped):
            code_lines.append(stripped)
        elif re.match(r"^\w+:\s*", stripped):  # labels
            code_lines.append(stripped)

    if code_lines:
        return "\n".join(code_lines)

    return query


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_WRITE_SYSTEM = """\
You are an expert Computer Architecture and Assembly Language tutor.
Write clear, well-commented assembly code for the student's request.

Rules:
- Default to MIPS assembly unless the student specifies another ISA.
- Include comments explaining each instruction.
- Show register usage conventions.
- If relevant, include .data and .text sections.
"""

_EXPLAIN_SYSTEM = """\
You are an expert Computer Architecture and Assembly Language tutor.
Explain the assembly concept, instruction, or code the student is asking about.
Use the textbook context if available.
Be clear and pedagogically helpful.
"""


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def code_agent_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node – Code specialist.

    Sub-tasks:
        trace   – Uses trace_assembly_code tool for step-by-step execution.
        write   – Uses LLM to generate assembly code.
        explain – Uses textbook + LLM (+ optional web) to explain code concepts.
    """
    query: str = state.get("current_query", "")
    conversation: str = state.get("conversation_context", "No previous conversation.")
    tool_log: list[str] = list(state.get("tool_calls_log", []))
    sources: list[dict] = list(state.get("sources", []))

    try:
        sub_task = _detect_code_sub_task(query)

        # ================================================================
        # SUB-TASK: TRACE
        # ================================================================
        if sub_task == "trace":
            code = _extract_code_block(query)
            trace_tool = _get_trace_assembly()
            trace_result = trace_tool.invoke(code)
            tool_log.append(f"trace_assembly_code(code=<{len(code)} chars>)")

            # Use LLM to add pedagogical commentary
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an assembly language tutor. The trace tool "
                        "has produced a step-by-step execution trace. Present "
                        "it clearly and add any helpful observations about "
                        "what the code accomplishes overall."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Student's code:\n```\n{code}\n```\n\n"
                        f"Trace output:\n{trace_result}\n\n"
                        "Present the trace with clear formatting and summarize "
                        "what the code does."
                    ),
                },
            ]
            answer = generate_with_chat_template(messages, max_new_tokens=1024)
            sources.append({"type": "code_trace"})

        # ================================================================
        # SUB-TASK: WRITE
        # ================================================================
        elif sub_task == "write":
            # Optionally look up ISA reference from textbook
            textbook_context = ""
            try:
                search_textbook = _get_search_textbook()
                textbook_context = search_textbook.invoke(query)
                tool_log.append(f"search_textbook(query={query!r})  [ISA reference]")
            except Exception:
                pass

            messages = [
                {"role": "system", "content": _WRITE_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Textbook reference:\n{textbook_context}\n\n"
                        f"Conversation context:\n{conversation}\n\n"
                        f"Student request: {query}\n\n"
                        "Write the assembly code with explanatory comments."
                    ),
                },
            ]
            answer = generate_with_chat_template(messages, max_new_tokens=1024)
            sources.append({"type": "code_generation"})

        # ================================================================
        # SUB-TASK: EXPLAIN
        # ================================================================
        else:
            # Gather context from textbook
            textbook_context = ""
            try:
                search_textbook = _get_search_textbook()
                textbook_context = search_textbook.invoke(query)
                tool_log.append(f"search_textbook(query={query!r})  [code reference]")
            except Exception:
                pass

            # Optional web supplement
            web_context = ""
            if len(textbook_context.strip()) < 80:
                try:
                    web_search = _get_web_search()
                    web_context = web_search.invoke(query)
                    tool_log.append(f"web_search(query={query!r})  [code reference fallback]")
                except Exception:
                    pass

            combined_context = textbook_context
            if web_context:
                combined_context += f"\n\n--- Web reference ---\n{web_context}"

            messages = [
                {"role": "system", "content": _EXPLAIN_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Reference material:\n{combined_context}\n\n"
                        f"Conversation context:\n{conversation}\n\n"
                        f"Student question: {query}\n\n"
                        "Provide a clear explanation."
                    ),
                },
            ]
            answer = generate_with_chat_template(messages, max_new_tokens=1024)
            sources.append({"type": "code_explanation"})

        print(f"💻 Code agent ({sub_task}) produced answer ({len(answer)} chars)")

        return {
            "agent_output": answer,
            "tool_calls_log": tool_log,
            "sources": sources,
        }

    except Exception as exc:
        error_msg = f"Code agent error: {exc}\n{traceback.format_exc()}"
        print(f"❌ {error_msg}")
        return {
            "agent_output": (
                "I'm sorry, I encountered an error while processing your code question. "
                "Please check the code syntax and try again."
            ),
            "tool_calls_log": tool_log,
            "sources": sources,
            "error": error_msg,
        }
