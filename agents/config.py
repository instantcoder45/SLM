"""
Central configuration for the Multi-Agent SLM system.
Auto-detects environment (Colab vs Local) and sets paths accordingly.
Supports two LLM backends: 'huggingface' (GPU/Colab) and 'ollama' (local CPU).
"""

import os
import shutil
from dataclasses import dataclass, field
from typing import Optional


def _detect_environment() -> str:
    """Detect if running in Google Colab or locally."""
    try:
        import google.colab  # noqa: F401
        # Set cache directories to Google Drive so models don't re-download every session
        os.environ["HF_HOME"] = "/content/drive/MyDrive/SLM/hf_cache"
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = "/content/drive/MyDrive/SLM/hf_cache"
        return "colab"
    except ImportError:
        return "local"


def _detect_device() -> str:
    """Detect best available device."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        try:
            if torch.backends.mps.is_available():
                return "mps"
        except AttributeError:
            pass
    except ImportError:
        pass
    return "cpu"


def _detect_backend() -> str:
    """
    Auto-detect the best LLM backend.
    - 'ollama' if Ollama is installed and we're on CPU (fast local inference)
    - 'huggingface' if CUDA is available (GPU inference)
    - 'huggingface' as fallback
    """
    device = _detect_device()

    # On GPU, always use HuggingFace (direct model loading)
    if device == "cuda":
        return "huggingface"

    # On CPU, prefer Ollama if installed (much faster with quantization)
    if shutil.which("ollama") is not None:
        return "ollama"

    # Fallback to HuggingFace (will work but slow on CPU)
    return "huggingface"


@dataclass
class Config:
    """Central configuration for the agent system."""

    # ----- LLM Backend -----
    # 'ollama' for local CPU (fast, quantized)
    # 'huggingface' for GPU/Colab (full precision)
    backend: str = field(default_factory=_detect_backend)

    # ----- LLM Settings -----
    # For HuggingFace backend:
    model_name: str = "microsoft/Phi-3.5-mini-instruct"
    # For Ollama backend:
    ollama_model: str = "phi3.5"
    ollama_base_url: str = "http://localhost:11434"

    max_new_tokens: int = 1024
    temperature: float = 0.1
    do_sample: bool = True

    # ----- Embedding Settings -----
    embed_model: str = "all-MiniLM-L6-v2"

    # ----- RAG Settings -----
    top_k: int = 5
    chapters_db_path: Optional[str] = None  # Auto-detected if None

    # ----- Memory Settings -----
    max_history: int = 10  # Number of conversation turns to remember

    # ----- Agent Settings -----
    max_agent_iterations: int = 3  # Max retry loops per query
    agent_timeout: int = 60  # Seconds before agent timeout

    # ----- Environment (auto-detected) -----
    environment: str = field(default_factory=_detect_environment)
    device: str = field(default_factory=_detect_device)

    def __post_init__(self):
        """Auto-detect chapters_db path based on environment."""
        if self.chapters_db_path is None:
            if self.environment == "colab":
                self.chapters_db_path = "/content/drive/My Drive/Colab_RAG_Project/chapters_db"
            else:
                # Local: relative to project root
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                self.chapters_db_path = os.path.join(project_root, "chapters_db")

        # Validate chapters_db exists
        if not os.path.exists(self.chapters_db_path):
            print(f"⚠️  Warning: chapters_db not found at {self.chapters_db_path}")
            print(f"   Run separate_chps.py first, or set config.chapters_db_path manually.")

    @property
    def torch_dtype(self):
        """Get appropriate torch dtype for the device."""
        import torch
        if self.device == "cuda":
            return torch.float16
        return torch.float32

    def summary(self) -> str:
        """Print a summary of the configuration."""
        model_display = self.ollama_model if self.backend == "ollama" else self.model_name.split('/')[-1]
        return (
            f"╔══════════════════════════════════════════╗\n"
            f"║        Agent System Configuration        ║\n"
            f"╠══════════════════════════════════════════╣\n"
            f"║  Environment : {self.environment:<25} ║\n"
            f"║  Backend     : {self.backend:<25} ║\n"
            f"║  Device      : {self.device:<25} ║\n"
            f"║  LLM         : {model_display:<25} ║\n"
            f"║  Embedder    : {self.embed_model:<25} ║\n"
            f"║  Top-K       : {self.top_k:<25} ║\n"
            f"║  Memory      : {self.max_history:<25} turns ║\n"
            f"║  DB Path     : {'.../' + os.path.basename(self.chapters_db_path):<25} ║\n"
            f"╚══════════════════════════════════════════╝"
        )
