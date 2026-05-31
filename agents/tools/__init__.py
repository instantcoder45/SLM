"""
LangChain tools for the Multi-Agent Computer Architecture Teaching Assistant.

Provides specialized tools that agents can invoke during reasoning:
- search_textbook: RAG retrieval over the textbook's FAISS indices
- calculator: Safe mathematical expression evaluator
- lookup_definition: Glossary of Computer Architecture terms
- summarize_chapter: LLM-powered chapter summarization
- compare_concepts: Structured comparison of two concepts
- trace_assembly_code: LLM-simulated MIPS assembly tracing
- web_search: DuckDuckGo web search for supplementary info
"""

from agents.tools.rag_tool import search_textbook, init_rag_tool
from agents.tools.calculator_tool import calculator
from agents.tools.glossary_tool import lookup_definition
from agents.tools.summarizer_tool import summarize_chapter, init_summarizer_tool
from agents.tools.comparison_tool import compare_concepts, init_comparison_tool
from agents.tools.code_exec_tool import trace_assembly_code
from agents.tools.web_search_tool import web_search

__all__ = [
    # Tools
    "search_textbook",
    "calculator",
    "lookup_definition",
    "summarize_chapter",
    "compare_concepts",
    "trace_assembly_code",
    "web_search",
    # Init functions
    "init_rag_tool",
    "init_summarizer_tool",
    "init_comparison_tool",
]
