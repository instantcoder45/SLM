"""
Gradio Chat UI for the Multi-Agent SLM System.
Works in both Google Colab and local environments.

Usage:
    python app.py              # Launch Gradio UI
    python app.py --share      # Launch with public URL (for Colab)
"""

import argparse
import gradio as gr
from agents.config import Config
from agents.graph import build_graph, run_query
from agents.memory import ConversationMemory


# ============================================================
#  Global State
# ============================================================
graph = None
memory = None
config = None


def initialize(cfg: Config = None):
    """Initialize the agent system."""
    global graph, memory, config

    config = cfg or Config()
    print(config.summary())

    print("\nBuilding agent graph (this loads the LLM — may take a minute)...\n")
    graph = build_graph(config)
    memory = ConversationMemory(max_turns=config.max_history)

    print("Agent system ready!\n")


# ============================================================
#  Chat Handler
# ============================================================
def chat_handler(user_message: str, chat_history: list):
    """
    Process a user message through the agent graph.
    Returns updated chat history for Gradio.
    """
    global graph, memory

    if not user_message.strip():
        return chat_history, ""

    # Run query through the agent graph
    try:
        result = run_query(graph, user_message, memory)

        response = result["response"]
        tool_log = result.get("tool_calls_log", [])
        sources = result.get("sources", [])
        agent_used = result.get("agent_used", "unknown")

        # Build formatted response
        formatted_response = ""

        # Agent badge
        agent_badges = {
            "rag_agent": "Textbook Agent",
            "math_agent": "Math Agent",
            "knowledge_agent": "Knowledge Agent",
            "code_agent": "Code Agent",
            "direct": "Direct Chat",
        }
        badge = agent_badges.get(agent_used, agent_used)
        formatted_response += f"**[{badge}]**\n\n"

        # Main response
        formatted_response += response

        # Tool usage trace (collapsible)
        if tool_log:
            formatted_response += "\n\n<details><summary>Tools Used</summary>\n\n"
            for log_entry in tool_log:
                formatted_response += f"- {log_entry}\n"
            formatted_response += "\n</details>"

        # Update Gradio chat history
        chat_history.append((user_message, formatted_response))

    except Exception as e:
        error_msg = f"**Error**: {str(e)}\n\nPlease try rephrasing your question."
        chat_history.append((user_message, error_msg))

    return chat_history, ""


def clear_handler():
    """Clear conversation history."""
    global memory
    if memory:
        memory.clear()
    return [], ""


# ============================================================
#  Gradio UI
# ============================================================
def create_ui():
    """Build the Gradio chat interface."""

    with gr.Blocks(
        title="SLM Agent — Computer Architecture TA",
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="blue",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
        ),
        css="""
        .gradio-container { max-width: 900px !important; margin: auto; }
        .agent-header {
            text-align: center;
            padding: 20px 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2em;
            font-weight: 700;
        }
        .description { text-align: center; color: #666; margin-bottom: 20px; }
        footer { display: none !important; }
        """
    ) as demo:

        gr.HTML("""
            <div class="agent-header">Computer Architecture Teaching Assistant</div>
            <div class="description">
                Powered by Phi-3.5 + RAG | Multi-Agent System with LangGraph<br>
                <em>Ask about Computer Architecture concepts, calculations, assembly code, and more!</em>
            </div>
        """)

        chatbot = gr.Chatbot(
            label="Chat",
            height=500,
            show_label=False,
            bubble_full_width=False,
            avatar_images=None,
        )

        with gr.Row():
            msg_input = gr.Textbox(
                placeholder="Ask a question about Computer Architecture...",
                label="Your Question",
                show_label=False,
                scale=9,
                container=False,
            )
            send_btn = gr.Button("Send", variant="primary", scale=1)

        with gr.Row():
            clear_btn = gr.Button("Clear History", variant="secondary")

        gr.Examples(
            examples=[
                "What is the difference between RISC and CISC?",
                "Calculate the speedup if we improve 40% of the program by 2x",
                "Define pipeline hazard.",
                "Compare cache and virtual memory.",
                "Write RISCV assembly code to add two numbers.",
                "What is Branch target Buffer?",
                "What is Amdahl's Law?",
            ],
            inputs=msg_input,
            label="Try these examples:",
        )

        # Event handlers
        msg_input.submit(
            chat_handler,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input],
        )
        send_btn.click(
            chat_handler,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input],
        )
        clear_btn.click(
            clear_handler,
            outputs=[chatbot, msg_input],
        )

    return demo


# ============================================================
#  Entry Point
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SLM Agent Chat UI")
    parser.add_argument("--share", action="store_true", help="Create public URL (for Colab)")
    parser.add_argument("--port", type=int, default=7860, help="Port number")
    parser.add_argument("--model", type=str, default=None, help="Override model name")
    parser.add_argument("--backend", type=str, default=None, choices=["huggingface", "ollama"], help="Override LLM backend (huggingface or ollama)")
    parser.add_argument("--ollama-model", type=str, default=None, help="Override Ollama model name (default: phi3.5)")
    parser.add_argument("--db-path", type=str, default=None, help="Override chapters_db path")
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

    # Initialize
    initialize(cfg)

    # Launch
    demo = create_ui()
    demo.launch(
        share=args.share,
        server_port=args.port,
        server_name="0.0.0.0" if args.share else "127.0.0.1",
    )
