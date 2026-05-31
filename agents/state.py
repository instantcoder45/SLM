"""
LangGraph state definition.
Defines the shared state that flows through the agent graph.
"""

from typing import TypedDict, Optional, Annotated
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """
    Shared state that flows through the LangGraph workflow.
    
    Fields:
        messages: Full conversation message history (accumulated via add_messages).
        current_query: The current user query being processed.
        selected_agent: Which specialist agent the supervisor routed to.
        agent_output: The output produced by the specialist agent.
        tool_calls_log: Log of tools used during this query (for UI display).
        sources: Source citations from RAG retrieval.
        error: Error message if something went wrong.
        iteration: Loop counter to prevent infinite agent loops.
        conversation_context: Formatted string of past conversation turns.
    """
    messages: Annotated[list[BaseMessage], add_messages]
    current_query: str
    selected_agent: str
    agent_output: str
    tool_calls_log: list[str]
    sources: list[dict]
    error: Optional[str]
    iteration: int
    conversation_context: str
