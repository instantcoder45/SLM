"""
Multi-Agent System for SLM (Small Language Model) RAG Project.

This package provides a multi-agent architecture using LangChain and LangGraph
to orchestrate specialized agents for Computer Architecture Q&A.
"""

from agents.config import Config
from agents.graph import build_graph

__all__ = ["Config", "build_graph"]
