"""
LLM loader module.
Supports two backends:
  - 'huggingface': Loads Phi-3.5 via HuggingFace transformers (GPU/Colab)
  - 'ollama': Connects to a local Ollama server (fast CPU inference)

Uses singleton pattern so the model is loaded only once.
"""

from agents.config import Config

# Singleton storage
_llm_instance = None
_tokenizer_instance = None
_raw_pipeline = None
_backend = None  # "huggingface" or "ollama"
_ollama_model = None
_ollama_base_url = None


# ===================================================================
#  PUBLIC API (used by all agents)
# ===================================================================

def load_llm(config: Config):
    """
    Load the LLM based on the configured backend.
    Uses singleton pattern — subsequent calls return the cached instance.
    """
    global _backend
    _backend = config.backend

    if _backend == "ollama":
        return _load_ollama(config)
    else:
        return _load_huggingface(config)


def get_llm():
    """Get the cached LLM instance. Raises if not loaded yet."""
    if _llm_instance is None:
        raise RuntimeError("LLM not loaded yet. Call load_llm(config) first.")
    return _llm_instance


def generate_with_chat_template(messages: list[dict], max_new_tokens: int = 512) -> str:
    """
    Generate a response using the model's chat template.
    Works with both Ollama and HuggingFace backends.

    Args:
        messages: List of {"role": "system"/"user"/"assistant", "content": "..."}
        max_new_tokens: Max tokens to generate

    Returns:
        Generated text string
    """
    if _backend == "ollama":
        return _generate_ollama(messages, max_new_tokens)
    else:
        return _generate_huggingface(messages, max_new_tokens)


# ===================================================================
#  OLLAMA BACKEND
# ===================================================================

def _load_ollama(config: Config):
    """Load/connect to Ollama."""
    global _llm_instance, _ollama_model, _ollama_base_url

    if _llm_instance is not None:
        return _llm_instance

    print("=" * 60)
    print(f"  Connecting to Ollama: {config.ollama_model}")
    print(f"  URL: {config.ollama_base_url}")
    print("=" * 60)

    _ollama_model = config.ollama_model
    _ollama_base_url = config.ollama_base_url

    # Test connection
    try:
        import httpx
        resp = httpx.get(f"{_ollama_base_url}/api/tags", timeout=5)
        models = [m["name"] for m in resp.json().get("models", [])]
        if any(_ollama_model in m for m in models):
            print(f"✅ Model '{_ollama_model}' found in Ollama")
        else:
            print(f"⚠️  Model '{_ollama_model}' not found. Available: {models}")
            print(f"   Run: ollama pull {_ollama_model}")
    except Exception as e:
        print(f"⚠️  Could not connect to Ollama at {_ollama_base_url}: {e}")
        print(f"   Make sure Ollama is running: ollama serve")

    # LangChain Ollama wrapper (for compatibility)
    try:
        from langchain_community.llms import Ollama
        _llm_instance = Ollama(
            model=_ollama_model,
            base_url=_ollama_base_url,
            temperature=config.temperature,
            num_predict=config.max_new_tokens,
        )
        print(f"✅ Ollama LLM ready\n")
    except ImportError:
        print("⚠️  langchain_community not available, using direct HTTP")
        _llm_instance = "ollama_direct"  # Marker for direct HTTP mode

    return _llm_instance


def _generate_ollama(messages: list[dict], max_new_tokens: int = 512) -> str:
    """Generate with Ollama's chat API."""
    import httpx

    try:
        resp = httpx.post(
            f"{_ollama_base_url}/api/chat",
            json={
                "model": _ollama_model,
                "messages": messages,
                "stream": False,
                "options": {
                    "num_predict": max_new_tokens,
                    "temperature": 0.1,
                },
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()
    except Exception as e:
        return f"[Ollama error: {e}]"


# ===================================================================
#  HUGGINGFACE BACKEND (original)
# ===================================================================

def _load_huggingface(config: Config):
    """Load Phi-3.5 via HuggingFace transformers."""
    global _llm_instance, _tokenizer_instance, _raw_pipeline
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    from langchain_huggingface import HuggingFacePipeline

    if _llm_instance is not None:
        return _llm_instance

    print("=" * 60)
    print(f"  Loading LLM: {config.model_name}")
    print(f"  Device: {config.device} | Dtype: {config.torch_dtype}")
    print("=" * 60)

    # Load tokenizer
    _tokenizer_instance = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True
    )

    # Ensure pad token is set
    if _tokenizer_instance.pad_token is None:
        _tokenizer_instance.pad_token = _tokenizer_instance.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=config.torch_dtype,
        device_map="auto" if config.device == "cuda" else None,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    if config.device != "cuda":
        model = model.to(config.device)

    model.eval()

    # Create HuggingFace pipeline
    _raw_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=_tokenizer_instance,
        max_new_tokens=config.max_new_tokens,
        do_sample=config.do_sample,
        temperature=config.temperature,
        return_full_text=False,
        pad_token_id=_tokenizer_instance.eos_token_id,
    )

    # Wrap in LangChain
    _llm_instance = HuggingFacePipeline(pipeline=_raw_pipeline)

    print(f"✅ LLM loaded successfully on {config.device}\n")
    return _llm_instance


def _generate_huggingface(messages: list[dict], max_new_tokens: int = 512) -> str:
    """Generate with HuggingFace pipeline using chat template."""
    if _raw_pipeline is None or _tokenizer_instance is None:
        raise RuntimeError("HuggingFace pipeline not loaded. Call load_llm(config) first.")

    prompt = _tokenizer_instance.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    output = _raw_pipeline(prompt, max_new_tokens=max_new_tokens)
    return output[0]["generated_text"].strip()


# ===================================================================
#  LEGACY ACCESSORS (kept for backward compatibility)
# ===================================================================

def get_tokenizer():
    """Get the cached tokenizer instance (HuggingFace only)."""
    if _tokenizer_instance is None:
        raise RuntimeError("Tokenizer not loaded (only available with HuggingFace backend).")
    return _tokenizer_instance


def get_raw_pipeline():
    """Get the raw HuggingFace pipeline (HuggingFace only)."""
    if _raw_pipeline is None:
        raise RuntimeError("Pipeline not loaded (only available with HuggingFace backend).")
    return _raw_pipeline
