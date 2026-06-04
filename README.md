# SLM — Multi-Agent Computer Architecture Teaching Assistant

A **Multi-Agent Retrieval-Augmented Generation (RAG)** system built with [LangGraph](https://github.com/langchain-ai/langgraph) and powered by [Phi-3.5-mini-instruct](https://huggingface.co/microsoft/Phi-3.5-mini-instruct). It serves as an intelligent Teaching Assistant for **Computer Architecture**, capable of answering conceptual questions, performing calculations, looking up definitions, tracing assembly code, and more — all grounded in lecture slide content via FAISS vector search.

---

## Key Features

- **Multi-Agent Routing** — A Supervisor agent classifies each query and routes it to the best specialist (RAG, Math, Knowledge, or Code).
- **RAG over Lecture Slides** — Semantic search across 39 indexed lecture decks using FAISS + SentenceTransformer embeddings.
- **Safe Math Engine** — AST-based expression evaluator for performance metrics (CPI, speedup, Amdahl's Law) with no unsafe `eval()`.
- **Built-in Glossary** — 55+ Computer Architecture terms with fuzzy matching and contextual suggestions.
- **Assembly Code Tracing** — Step-by-step simulated execution of MIPS/RISC-V assembly with register state tracking.
- **Concept Comparison** — Structured side-by-side comparisons of architecture concepts grounded in lecture content.
- **Lecture Summarization** — On-demand summaries of any indexed lecture with caching.
- **Web Search Fallback** — DuckDuckGo search for supplementary information when textbook content is insufficient.
- **Conversation Memory** — Sliding-window memory (10 turns) for multi-turn contextual conversations.
- **Dual Interface** — Interactive CLI chat and Gradio web UI.
- **Dual Backend** — Supports both HuggingFace Transformers (GPU/Colab) and Ollama (fast local CPU inference).

---

## Architecture

```
START ──> Supervisor ──┬──> RAG Agent ──────────┐
                       ├──> Math Agent ─────────┤
                       ├──> Knowledge Agent ────┤
                       ├──> Code Agent ─────────┤──> END
                       └──> Direct Reply ───────┘
```

The **Supervisor** uses a 3-tier routing strategy:
1. **Regex** — Instant detection of greetings/chitchat → `DIRECT`
2. **LLM Classification** — Sends the query to the LLM with a router prompt to pick a specialist
3. **Keyword Fallback** — If the LLM fails to produce a clean answer, regex keyword patterns are used

Each specialist agent has access to its own set of **LangChain tools** and formats its output via the LLM before returning to the user.

---

## Project Structure

```
SLM/
├── main.py                  # CLI entry point (terminal chat, single query, or Gradio UI)
├── app.py                   # Gradio web UI entry point
├── RefineSlides.py          # Colab script: downloads & indexes lecture slide PDFs
├── final.ipynb              # Colab notebook: full system demo with test queries
│
├── agents/                  # Core multi-agent package
│   ├── __init__.py          # Package exports: Config, build_graph
│   ├── config.py            # Central configuration (auto-detects env, device, backend)
│   ├── graph.py             # LangGraph workflow: builds & compiles the agent graph
│   ├── llm.py               # LLM loader (HuggingFace / Ollama) with singleton pattern
│   ├── memory.py            # Sliding-window conversation memory
│   ├── state.py             # LangGraph shared state (TypedDict) definition
│   │
│   ├── agents/              # Specialist agent nodes
│   │   ├── supervisor.py    # Supervisor / Router — classifies & dispatches queries
│   │   ├── rag_agent.py     # RAG Agent — textbook search + grounded generation
│   │   ├── math_agent.py    # Math Agent — calculations & formula application
│   │   ├── knowledge_agent.py  # Knowledge Agent — definitions, summaries, comparisons
│   │   └── code_agent.py    # Code Agent — assembly tracing, writing, & explaining
│   │
│   └── tools/               # LangChain tools used by agents
│       ├── rag_tool.py      # FAISS semantic search across lecture indices
│       ├── calculator_tool.py   # Safe AST-based math expression evaluator
│       ├── glossary_tool.py     # 55+ term glossary with fuzzy matching
│       ├── code_exec_tool.py    # LLM-simulated assembly code tracing
│       ├── comparison_tool.py   # Structured concept comparison generator
│       ├── summarizer_tool.py   # Lecture summarization with caching
│       ├── lecture_registry.py  # Keyword-embedding registry for lecture relevance
│       └── web_search_tool.py   # DuckDuckGo fallback search
│
├── slides_db/               # Pre-built FAISS vector databases (one per lecture)
│   ├── Lecture-1/           # Each contains: index.faiss, texts.pkl, meta.pkl
│   ├── Lecture-2/
│   ├── ...
│   └── RARS/
│
└── prev/                    # Legacy / predecessor scripts
    ├── book.pdf             # Source textbook PDF
    ├── separate_chps.py     # PDF → chapter splitting, chunking, & FAISS indexing
    ├── clean_text.py        # Regex-based PDF text cleaning (100+ artifact fix patterns)
    ├── retriever.py         # Monolithic single-agent RAG prototype (predecessor)
    └── finetuning_approach/ # Fine-tuning experiments
        ├── finetuning_dataset_creation_and_cleaning.ipynb
        ├── TrainingSLM.ipynb
        └── TrainingSLM_with_agent.ipynb
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/) (recommended for local use) **or** a CUDA-capable GPU

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd SLM

# Install dependencies
pip install langchain langgraph langchain-community langchain-huggingface \
            transformers sentence-transformers faiss-cpu \
            gradio duckduckgo-search httpx tiktoken
```

### Running Locally (with Ollama — Recommended)

```bash
# 1. Install and start Ollama
ollama serve

# 2. Pull the Phi-3.5 model
ollama pull phi3.5

# 3. Launch the terminal chat
python main.py

# Or launch the Gradio web UI
python main.py --ui
```

### Running on Google Colab

Open **`final.ipynb`** in Colab. It will:
1. Mount Google Drive
2. Install all dependencies
3. Load the Phi-3.5-mini-instruct model on GPU
4. Build the agent graph and start an interactive chat

### CLI Options

```
python main.py [OPTIONS]

Options:
  --ui                Launch Gradio web UI instead of terminal chat
  --query "question"  Run a single query and exit
  --model MODEL       Override HuggingFace model name
  --backend TYPE      Force backend: 'huggingface' or 'ollama'
  --ollama-model NAME Override Ollama model name (default: phi3.5)
  --db-path PATH      Override path to the lecture vector databases
  --share             Create a public Gradio URL (for Colab)
```

---

## Agents in Detail

### Supervisor
Routes queries using a hybrid strategy (regex → LLM → keyword fallback). Detects greetings, arithmetic expressions, code requests, definition lookups, and textbook questions.

### RAG Agent
Searches the FAISS lecture indices using the **Lecture Registry** (cosine similarity over keyword embeddings) to focus retrieval on the most relevant lectures. Generates answers grounded in retrieved passages with source citations. Falls back to web search if textbook content is thin.

### Math Agent
Handles two types of math queries:
- **Pure arithmetic** (e.g., `2^10`) — directly evaluated via the safe calculator.
- **Applied problems** (e.g., Amdahl's Law) — searches the textbook for relevant formulas, asks the LLM to set up `CALC: <expr>` expressions, evaluates them, and produces an explained answer.

### Knowledge Agent
Sub-routes internally to three tools:
- **Define** — Looks up terms in the built-in glossary (fuzzy matching with suggestions).
- **Summarize** — Generates a structured summary of any indexed lecture.
- **Compare** — Produces a side-by-side comparison table grounded in lecture content.

### Code Agent
Handles assembly language tasks:
- **Trace** — Simulates step-by-step execution of MIPS/RISC-V code with register state tracking.
- **Write** — Generates commented assembly code, optionally referencing ISA details from the textbook.
- **Explain** — Explains assembly/architecture concepts using textbook + web search context.

---

## Tools Reference

| Tool | Used By | Description |
|------|---------|-------------|
| `search_textbook` | RAG, Math, Code | FAISS semantic search across 39 lecture vector databases |
| `calculator` | Math | Safe AST-based evaluator supporting arithmetic, sqrt, log, trig |
| `lookup_definition` | Knowledge | Glossary lookup with fuzzy matching (55+ terms) |
| `summarize_chapter` | Knowledge | On-demand lecture summarization with caching |
| `compare_concepts` | Knowledge | Structured side-by-side concept comparison |
| `trace_assembly_code` | Code | LLM-simulated step-by-step assembly execution |
| `web_search` | RAG, Code | DuckDuckGo fallback search (top 3 results) |

---

## Data Pipeline

The `slides_db/` directory contains pre-built FAISS vector databases created by **`RefineSlides.py`**:

1. **Download** lecture slide PDFs from Google Drive
2. **Extract** text from each slide using `pdfplumber`
3. **Clean** text (unicode normalization, artifact removal)
4. **Extract keywords** via TF-IDF (top 30 per lecture)
5. **Embed** slide text chunks using `all-MiniLM-L6-v2` (SentenceTransformer)
6. **Index** embeddings in per-lecture FAISS indices

Each lecture folder in `slides_db/` contains:
- `index.faiss` — Vector index for similarity search
- `texts.pkl` — Raw text chunks corresponding to vectors
- `meta.pkl` — Metadata including lecture name and TF-IDF keywords

---

## Legacy Scripts (`prev/`)

The `prev/` directory contains the earlier iteration of this project:

| File | Description |
|------|-------------|
| `separate_chps.py` | Splits the textbook PDF into chapters, cleans text, builds per-chapter FAISS vector databases |
| `clean_text.py` | 100+ regex patterns to fix PDF extraction artifacts (broken words, garbage chars, headers/footers) |
| `retriever.py` | Monolithic single-agent RAG system — the predecessor to the current multi-agent architecture |
| `finetuning_approach/` | Notebooks documenting initial fine-tuning experiments on Phi-3 (exploratory, not integrated into the current system) |

---

## Configuration

The system auto-detects its environment at startup:

| Setting | Colab | Local (GPU) | Local (CPU) |
|---------|-------|-------------|-------------|
| Backend | HuggingFace | HuggingFace | Ollama (if installed) |
| Device | CUDA | CUDA | CPU |
| LLM | Phi-3.5-mini-instruct | Phi-3.5-mini-instruct | phi3.5 (quantized) |
| Embedder | all-MiniLM-L6-v2 | all-MiniLM-L6-v2 | all-MiniLM-L6-v2 |
| DB Path | Google Drive | `./chapters_db` | `./chapters_db` |

All settings can be overridden via CLI flags or by modifying `agents/config.py`.

---

## Related Work and Future Prospects

### LoRA Fine-Tuning
- Fine-tune the base Phi-3.5 model using **LoRA (Low-Rank Adaptation)** on a high-quality Question-Answer dataset curated from Computer Architecture content.
- This would improve instruction-following, domain accuracy, and reduce hallucinations compared to the current prompt-only approach.

### Improved RAG Pipeline
- **Hybrid Search** — Combine keyword-based (BM25) and vector-based retrieval for better recall.
- **Cross-Encoder Re-ranking** — Re-rank retrieved chunks using a cross-encoder model to improve precision before passing context to the LLM.

### Interface Improvements
- **Streaming Responses** — Implement token-by-token streaming in both the CLI and Gradio UI to provide real-time feedback instead of waiting for the full response.
- **Source Previews** — Display expandable source previews alongside answers so students can read the original lecture content directly.
- **Export Chat History** — Allow students to download conversation transcripts as PDF or Markdown for study reference.

### Response Speed Optimization
- **Model Quantization** — Deploy 4-bit or 8-bit quantized models (GPTQ/AWQ) to significantly reduce inference latency on both CPU and GPU.
- **Async Tool Execution** — Run independent tool calls (e.g., textbook search + glossary lookup) in parallel rather than sequentially.
- **Embedding Cache** — Cache frequently queried embeddings to skip redundant encoding on repeated or similar questions.

### Enhanced Tooling
- Advanced math/unit converters and graphing/plotting tools.
- External glossary and documentation lookups beyond the built-in term database.
