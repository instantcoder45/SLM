"""
CLI entry point for the Multi-Agent SLM System.

Usage:
    python main.py                        # Interactive chat in terminal
    python main.py --ui                   # Launch Gradio web UI
    python main.py --query "question"     # Single query mode
"""

import argparse
import sys

from agents.config import Config
from agents.graph import build_graph, run_query
from agents.memory import ConversationMemory


def interactive_chat(config: Config):
    """Run an interactive chat loop in the terminal."""

    print(config.summary())
    print("\n⏳ Building agent graph (loading LLM — may take a minute)...\n")

    graph = build_graph(config)
    memory = ConversationMemory(max_turns=config.max_history)

    print("=" * 60)
    print("  🤖 Computer Architecture Teaching Assistant")
    print("  Multi-Agent System with LangGraph")
    print("=" * 60)
    print("  Commands:")
    print("    exit    — Quit")
    print("    clear   — Clear conversation history")
    print("    history — Show conversation history")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("📝 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye! 👋")
            break

        if not user_input:
            continue

        if user_input.lower() == "exit":
            print("Goodbye! 👋")
            break

        if user_input.lower() == "clear":
            memory.clear()
            print("🗑️  Conversation history cleared.\n")
            continue

        if user_input.lower() == "history":
            ctx = memory.get_context_string()
            print(f"\n📜 Conversation History ({memory.turn_count} turns):\n{ctx}\n")
            continue

        # Run query
        print()
        try:
            result = run_query(graph, user_input, memory)

            response = result["response"]
            tool_log = result.get("tool_calls_log", [])
            sources = result.get("sources", [])
            agent_used = result.get("agent_used", "unknown")

            # Agent badge
            agent_names = {
                "rag_agent": "📚 Textbook Agent",
                "math_agent": "🧮 Math Agent",
                "knowledge_agent": "📖 Knowledge Agent",
                "code_agent": "💻 Code Agent",
                "direct": "💬 Direct Chat",
            }
            agent_label = agent_names.get(agent_used, agent_used)

            print(f"[{agent_label}]")
            print(f"🤖 Assistant: {response}")

            if sources:
                print("\n📖 Sources:")
                for src in sources:
                    chapter = src.get("chapter", "?")
                    score = src.get("distance", 0)
                    print(f"   - {chapter} (relevance: {1/(1+score):.0%})")

            if tool_log:
                print("\n🔧 Tools used:", ", ".join(tool_log))



        except Exception as e:
            print(f"❌ Error: {e}\n")

        print("-" * 50 + "\n")


def single_query(config: Config, query: str):
    """Run a single query and print the result."""

    print(config.summary())
    print(f"\n⏳ Building agent graph...\n")

    graph = build_graph(config)
    memory = ConversationMemory(max_turns=1)

    result = run_query(graph, query, memory)

    print(f"Query: {query}")
    print(f"Agent: {result.get('agent_used', 'unknown')}")
    print(f"Response: {result['response']}")

    if result.get("sources"):
        print("\nSources:")
        for src in result["sources"]:
            print(f"  - {src.get('chapter', '?')}")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Agent Computer Architecture Teaching Assistant"
    )
    parser.add_argument(
        "--ui", action="store_true",
        help="Launch Gradio web UI instead of terminal chat"
    )
    parser.add_argument(
        "--query", type=str, default=None,
        help="Run a single query and exit"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Override model name (e.g., 'HuggingFaceTB/SmolLM3-3B' for testing)"
    )
    parser.add_argument(
        "--backend", type=str, default=None, choices=["huggingface", "ollama"],
        help="Override LLM backend (huggingface or ollama)"
    )
    parser.add_argument(
        "--ollama-model", type=str, default=None,
        help="Override Ollama model name (default: phi3.5)"
    )
    parser.add_argument(
        "--db-path", type=str, default=None,
        help="Override path to chapters_db"
    )
    parser.add_argument(
        "--share", action="store_true",
        help="Create public URL for Gradio (use with --ui)"
    )
    args = parser.parse_args()

    # Build config
    cfg = Config()
    if args.model:
        cfg.model_name = args.model
    if args.backend:
        cfg.backend = args.backend
    if args.ollama_model:
        cfg.ollama_model = args.ollama_model
    if args.db_path:
        cfg.chapters_db_path = args.db_path

    # Route to appropriate mode
    if args.ui:
        from app import initialize, create_ui
        initialize(cfg)
        demo = create_ui()
        demo.launch(share=args.share, server_name="0.0.0.0" if args.share else "127.0.0.1")
    elif args.query:
        single_query(cfg, args.query)
    else:
        interactive_chat(cfg)


if __name__ == "__main__":
    main()
