"""
Main LangGraph workflow orchestration.

Builds the multi-agent graph, wires up conditional routing from the
supervisor to specialist agents, and provides a convenience ``run_query``
helper for interactive use.

Graph topology::

    START ──▶ supervisor ──┬──▶ rag_agent ──────┐
                           ├──▶ math_agent ─────┤
                           ├──▶ knowledge_agent ┤
                           ├──▶ code_agent ─────┤──▶ END
                           └──▶ direct_reply ───┘
"""

from __future__ import annotations

import warnings
import traceback
from typing import Any

# Suppress noisy deprecation warnings from Jupyter / transformers
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*seen_tokens.*")
warnings.filterwarnings("ignore", message=".*flash-attention.*")

from langgraph.graph import StateGraph, END

from agents.state import AgentState
from agents.config import Config
from agents.llm import generate_with_chat_template

# ---------------------------------------------------------------------------
# Lazy agent imports (avoids import-time side-effects)
# ---------------------------------------------------------------------------

from agents.agents.supervisor import supervisor_node
from agents.agents.rag_agent import rag_agent_node
from agents.agents.math_agent import math_agent_node
from agents.agents.knowledge_agent import knowledge_agent_node
from agents.agents.code_agent import code_agent_node


# ---------------------------------------------------------------------------
# Direct reply node – handles greetings / chitchat (no specialist needed)
# ---------------------------------------------------------------------------

_DIRECT_SYSTEM = """\
You are a friendly Computer Architecture teaching assistant.
Respond naturally to the student's message.  Keep it brief and helpful.
If they greet you, greet them back and offer to help with computer
architecture topics.
"""


def direct_reply_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node – handles DIRECT-routed queries (greetings, chitchat).
    This is the only node that makes an LLM call for non-specialist queries.
    """
    try:
        query = state.get("current_query", "")
        messages = [
            {"role": "system", "content": _DIRECT_SYSTEM},
            {"role": "user", "content": query},
        ]
        answer = generate_with_chat_template(messages, max_new_tokens=256)
        return {"agent_output": answer}
    except Exception as exc:
        print(f"❌ Direct reply error: {exc}")
        return {"agent_output": "Hello! I'm your Computer Architecture teaching assistant. How can I help you today?"}


# ---------------------------------------------------------------------------
# Routing function for conditional edges
# ---------------------------------------------------------------------------

def _route_from_supervisor(state: AgentState) -> str:
    """
    Conditional-edge function.

    Maps ``state['selected_agent']`` to the corresponding graph node name.
    """
    selected = state.get("selected_agent", "DIRECT")
    mapping = {
        "RAG_AGENT": "rag_agent",
        "MATH_AGENT": "math_agent",
        "KNOWLEDGE_AGENT": "knowledge_agent",
        "CODE_AGENT": "code_agent",
        "DIRECT": "direct_reply",
    }
    return mapping.get(selected, "rag_agent")


# ---------------------------------------------------------------------------
# Tool initialisation helpers
# ---------------------------------------------------------------------------

def _init_tools(config: Config) -> None:
    """
    Call init functions on all tools that require setup (e.g. loading
    FAISS indices, embedding models, glossary data).

    Each tool module exposes an optional ``init_*`` function.  Missing
    init functions are silently skipped.
    """
    init_functions = [
        ("agents.tools", "init_rag_tool"),
        ("agents.tools", "init_summarizer_tool"),
        ("agents.tools", "init_comparison_tool"),
    ]

    for module_path, func_name in init_functions:
        try:
            import importlib
            mod = importlib.import_module(module_path)
            init_fn = getattr(mod, func_name, None)
            if init_fn is not None:
                init_fn(config)
                print(f"  ✅ {func_name}() initialised")
            else:
                print(f"  ⏭️  {func_name}() not found – skipped")
        except Exception as exc:
            print(f"  ⚠️  {func_name}() failed: {exc}")


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(config: Config):
    """
    Build and compile the multi-agent LangGraph workflow.

    Steps:
        1. Load the LLM (singleton; safe to call multiple times).
        2. Initialise tool backends (FAISS, glossary, etc.).
        3. Construct the ``StateGraph`` with all nodes and edges.
        4. Return the compiled graph.

    Args:
        config: Central ``Config`` dataclass instance.

    Returns:
        A compiled LangGraph ``CompiledGraph`` ready for ``.invoke()``.
    """
    # ---- 1. Load LLM ----
    from agents.llm import load_llm
    print("🔧 Building agent graph …")
    load_llm(config)

    # ---- 2. Init tools ----
    print("🔧 Initialising tools …")
    _init_tools(config)

    # ---- 3. Build graph ----
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("rag_agent", rag_agent_node)
    graph.add_node("math_agent", math_agent_node)
    graph.add_node("knowledge_agent", knowledge_agent_node)
    graph.add_node("code_agent", code_agent_node)
    graph.add_node("direct_reply", direct_reply_node)

    # Entry point
    graph.set_entry_point("supervisor")

    # Conditional routing from supervisor
    graph.add_conditional_edges(
        "supervisor",
        _route_from_supervisor,
        {
            "rag_agent": "rag_agent",
            "math_agent": "math_agent",
            "knowledge_agent": "knowledge_agent",
            "code_agent": "code_agent",
            "direct_reply": "direct_reply",
        },
    )

    # All nodes go directly to END (no synthesizer pass)
    for node_name in ["rag_agent", "math_agent", "knowledge_agent", "code_agent", "direct_reply"]:
        graph.add_edge(node_name, END)

    # ---- 4. Compile ----
    compiled = graph.compile()
    print("✅ Agent graph compiled successfully\n")
    return compiled


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

def run_query(
    graph,
    query: str,
    memory=None,
) -> dict[str, Any]:
    """
    Run a single query through the compiled agent graph.

    Args:
        graph:  Compiled LangGraph (from ``build_graph``).
        query:  The student's question.
        memory: Optional ``ConversationMemory`` instance for context.

    Returns:
        Dict with keys: response, tool_calls_log, sources, agent_used.
    """
    from langchain_core.messages import HumanMessage

    # Build conversation context from memory
    conversation_context = "No previous conversation."
    if memory is not None:
        conversation_context = memory.get_context_string()

    # Assemble initial state
    initial_state: dict[str, Any] = {
        "messages": [HumanMessage(content=query)],
        "current_query": query,
        "selected_agent": "",
        "agent_output": "",
        "tool_calls_log": [],
        "sources": [],
        "error": None,
        "iteration": 0,
        "conversation_context": conversation_context,
    }

    # Invoke the graph
    try:
        final_state = graph.invoke(initial_state)
    except Exception as exc:
        error_msg = f"Graph execution error: {exc}\n{traceback.format_exc()}"
        print(f"❌ {error_msg}")
        return {
            "response": "I'm sorry, an internal error occurred. Please try again.",
            "tool_calls_log": [f"ERROR: {exc}"],
            "sources": [],
            "agent_used": "error",
        }

    # Extract results
    response = final_state.get("agent_output", "No response generated.")
    tool_log = final_state.get("tool_calls_log", [])
    sources = final_state.get("sources", [])
    agent_used = final_state.get("selected_agent", "unknown")

    # Map agent names to lowercase for UI display
    agent_map = {
        "RAG_AGENT": "rag_agent",
        "MATH_AGENT": "math_agent",
        "KNOWLEDGE_AGENT": "knowledge_agent",
        "CODE_AGENT": "code_agent",
        "DIRECT": "direct",
    }
    agent_used = agent_map.get(agent_used, agent_used.lower() if agent_used else "unknown")

    # Update memory if provided
    if memory is not None:
        memory.add_user_message(query)
        memory.add_assistant_message(response)

    return {
        "response": response,
        "tool_calls_log": tool_log,
        "sources": sources,
        "agent_used": agent_used,
    }
