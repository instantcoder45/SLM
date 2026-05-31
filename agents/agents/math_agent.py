"""
Math specialist agent.

Handles numeric calculations, performance-metric problems (CPI, speedup,
Amdahl's Law, etc.), and applied textbook math.  Combines formula lookup
from the textbook with a safe calculator tool.
"""

from __future__ import annotations

import traceback
from typing import Any

from agents.state import AgentState
from agents.llm import generate_with_chat_template

# ---------------------------------------------------------------------------
# Lazy tool imports
# ---------------------------------------------------------------------------

def _get_search_textbook():
    from agents.tools import search_textbook
    return search_textbook

def _get_calculator():
    from agents.tools import calculator
    return calculator


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_FORMULA_SYSTEM = """\
You are a Computer Architecture math tutor.  Given a student's question
and any relevant textbook context, do the following:

1. Identify the formula(s) needed.
2. Show the formula with variable names.
3. Substitute the given values.
4. Provide the EXACT arithmetic expression(s) that must be evaluated.
   Format each expression on its own line prefixed with "CALC: ".
   Example: CALC: (5 * 3 + 2) / 10
5. After the CALC lines, add a line "EXPLANATION:" followed by a short
   explanation of what the result means in context.

If no calculation is needed, just answer directly and omit CALC lines.
"""

_FINAL_SYSTEM = """\
You are a Computer Architecture math tutor.  The student asked a question,
you found the relevant formula and computed the result.  Now present a
clear, step-by-step explanation that includes:
- The formula used
- The substituted values
- The computed result
- What the result means in context

Be concise but thorough.
"""


def _is_pure_math(query: str) -> bool:
    """Heuristic: does the query look like a raw arithmetic expression?"""
    import re
    # Strip whitespace and check if it's mostly math characters
    cleaned = query.strip()
    return bool(re.fullmatch(r"[\d\s\+\-\*/\(\)\.\^%]+", cleaned))


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def math_agent_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node – Math specialist.

    1. For pure arithmetic (e.g. "2+2"), evaluates directly via calculator.
    2. For applied problems, first searches the textbook for relevant formulas,
       then asks the LLM to set up the calculation, evaluates it, and
       produces a final explained answer.
    """
    query: str = state.get("current_query", "")
    conversation: str = state.get("conversation_context", "No previous conversation.")
    tool_log: list[str] = list(state.get("tool_calls_log", []))
    sources: list[dict] = list(state.get("sources", []))

    try:
        calc = _get_calculator()

        # ---- Fast path: pure arithmetic ----
        if _is_pure_math(query):
            result = calc.invoke(query)
            tool_log.append(f"calculator(expression={query!r})")
            answer = f"**Result:** `{query}` = **{result}**"
            print(f"🧮 Math agent (pure arithmetic): {query} = {result}")
            return {
                "agent_output": answer,
                "tool_calls_log": tool_log,
                "sources": sources,
            }

        # ---- Applied math problem ----

        # Step 1: Look up relevant formulas from the textbook
        textbook_context = ""
        try:
            search_textbook = _get_search_textbook()
            textbook_context = search_textbook.invoke(query)
            tool_log.append(f"search_textbook(query={query!r})  [formula lookup]")
        except Exception:
            textbook_context = "(No textbook context available.)"

        # Step 2: Ask LLM to set up the calculation
        setup_messages = [
            {"role": "system", "content": _FORMULA_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Textbook reference:\n{textbook_context}\n\n"
                    f"Conversation context:\n{conversation}\n\n"
                    f"Student question: {query}"
                ),
            },
        ]
        setup_output = generate_with_chat_template(setup_messages, max_new_tokens=400)

        # Step 3: Extract and evaluate CALC expressions
        import re
        calc_lines = re.findall(r"CALC:\s*(.+)", setup_output)
        calc_results: dict[str, str] = {}

        for expr in calc_lines:
            expr = expr.strip()
            try:
                result = calc.invoke(expr)
                calc_results[expr] = result
                tool_log.append(f"calculator(expression={expr!r})")
            except Exception as calc_err:
                calc_results[expr] = f"Error: {calc_err}"

        # Step 4: Build final explained answer
        if calc_results:
            results_text = "\n".join(
                f"  {expr} = {res}" for expr, res in calc_results.items()
            )
            final_messages = [
                {"role": "system", "content": _FINAL_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Student question: {query}\n\n"
                        f"Formula setup & reasoning:\n{setup_output}\n\n"
                        f"Computed results:\n{results_text}\n\n"
                        "Now present the complete, explained answer."
                    ),
                },
            ]
            answer = generate_with_chat_template(final_messages, max_new_tokens=1024)
        else:
            # LLM didn't produce CALC lines – the setup_output itself is the answer
            answer = setup_output

        print(f"🧮 Math agent produced answer ({len(answer)} chars, {len(calc_results)} calculations)")

        return {
            "agent_output": answer,
            "tool_calls_log": tool_log,
            "sources": sources,
        }

    except Exception as exc:
        error_msg = f"Math agent error: {exc}\n{traceback.format_exc()}"
        print(f"❌ {error_msg}")
        return {
            "agent_output": (
                "I'm sorry, I encountered an error while performing the calculation. "
                "Please check the expression and try again."
            ),
            "tool_calls_log": tool_log,
            "sources": sources,
            "error": error_msg,
        }
