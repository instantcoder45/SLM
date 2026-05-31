"""
Conversation memory manager.
Provides sliding-window memory for multi-turn conversations.
"""

from langchain_core.messages import HumanMessage, AIMessage
from typing import Optional


class ConversationMemory:
    """
    Simple sliding-window conversation memory.
    Stores the last N turns of conversation as LangChain message objects.
    """

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self._history: list[dict] = []  # [{"role": "user"/"assistant", "content": "..."}]

    def add_user_message(self, content: str):
        """Add a user message to history."""
        self._history.append({"role": "user", "content": content})
        self._trim()

    def add_assistant_message(self, content: str):
        """Add an assistant message to history."""
        self._history.append({"role": "assistant", "content": content})
        self._trim()

    def _trim(self):
        """Keep only the last max_turns * 2 messages (each turn = user + assistant)."""
        max_messages = self.max_turns * 2
        if len(self._history) > max_messages:
            self._history = self._history[-max_messages:]

    def get_history(self) -> list[dict]:
        """Get conversation history as list of dicts."""
        return list(self._history)

    def get_langchain_messages(self):
        """Get history as LangChain message objects."""
        messages = []
        for msg in self._history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
        return messages

    def get_context_string(self) -> str:
        """Get history as a formatted string for prompt injection."""
        if not self._history:
            return "No previous conversation."

        parts = []
        for msg in self._history:
            role = "Student" if msg["role"] == "user" else "Assistant"
            parts.append(f"{role}: {msg['content']}")
        return "\n".join(parts)

    def clear(self):
        """Clear all conversation history."""
        self._history.clear()

    @property
    def turn_count(self) -> int:
        """Number of complete turns (user+assistant pairs)."""
        return len(self._history) // 2

    def __len__(self) -> int:
        return len(self._history)

    def __repr__(self) -> str:
        return f"ConversationMemory(turns={self.turn_count}, max={self.max_turns})"
