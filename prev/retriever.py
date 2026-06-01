import os
import pickle
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# ================= CONFIG =================
CHAPTERS_DB = "chapters_db"
EMBED_MODEL = "all-MiniLM-L6-v2"  # Same as used in separate_chps.py
TOP_K = 5  # Number of chunks to retrieve

# SmolLM model (smaller, faster for testing)
MODEL_NAME = "HuggingFaceTB/SmolLM3-3B"

# Phi-3.5 for Colab (uncomment when using GPU)
# MODEL_NAME = "unsloth/Phi-3.5-mini-instruct"
# ==========================================

embedder = SentenceTransformer(EMBED_MODEL)


def load_all_chapters():
    """Load FAISS indexes and texts from all chapters."""
    chapters = {}
    
    for chapter_dir in os.listdir(CHAPTERS_DB):
        chapter_path = os.path.join(CHAPTERS_DB, chapter_dir)
        if not os.path.isdir(chapter_path):
            continue
        
        index_path = os.path.join(chapter_path, "index.faiss")
        texts_path = os.path.join(chapter_path, "texts.pkl")
        meta_path = os.path.join(chapter_path, "meta.pkl")
        
        if not all(os.path.exists(p) for p in [index_path, texts_path, meta_path]):
            continue
        
        index = faiss.read_index(index_path)
        with open(texts_path, "rb") as f:
            texts = pickle.load(f)
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        
        chapters[chapter_dir] = {
            "index": index,
            "texts": texts,
            "meta": meta
        }
    
    print(f"✅ Loaded {len(chapters)} chapters")
    return chapters


def retrieve(query, chapters, top_k=TOP_K):
    """
    Retrieve most relevant chunks across all chapters.
    Returns list of (chapter_name, chunk_text, distance).
    """
    query_embedding = embedder.encode([query]).astype("float32")
    
    all_results = []
    
    for chapter_name, chapter_data in chapters.items():
        index = chapter_data["index"]
        texts = chapter_data["texts"]
        
        # Search in this chapter's index
        k = min(top_k, len(texts))
        distances, indices = index.search(query_embedding, k)
        
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(texts):
                all_results.append({
                    "chapter": chapter_name,
                    "text": texts[idx],
                    "distance": float(dist)
                })
    
    # Sort by distance (lower = more similar for L2)
    all_results.sort(key=lambda x: x["distance"])
    
    return all_results[:top_k]


def build_context(retrieved_chunks):
    """Build context string from retrieved chunks."""
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        context_parts.append(
            f"[Source: {chunk['chapter']}]\n{chunk['text']}"
        )
    return "\n\n---\n\n".join(context_parts)


def load_model():
    """Load LLM model (downloads automatically on first run)."""
    print("=" * 60)
    print(f"LOADING {MODEL_NAME}")
    print("=" * 60)
    print("• First run will download the model")
    print("• Later runs load from HuggingFace cache\n")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    ).to(device)
    
    model.eval()
    print(f"✓ Model loaded on {device}\n")
    return model, tokenizer


@torch.no_grad()
def generate(model, tokenizer, prompt):
    """Generate response using Phi-3.5."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer based on the given context."},
        {"role": "user", "content": prompt}
    ]
    
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    output = model.generate(
        **inputs,
        max_new_tokens=512, #increase max new tokens for longer output
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id
    )
    
    # Decode only newly generated tokens
    generated_ids = output[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return response.strip()


def rag_generate(model, tokenizer, query, chapters):
    """
    Full RAG pipeline:
    1. Retrieve relevant chunks
    2. Build context
    3. Generate answer with LLM
    """
    # Retrieve
    retrieved = retrieve(query, chapters)
    context = build_context(retrieved)
    
    # Build RAG prompt
    rag_prompt = f"""Use the following context to answer the question. If the answer is not in the context, say so.

Context:
{context}

Question: {query}

Answer:"""
    
    # Generate
    response = generate(model, tokenizer, rag_prompt)
    
    return response, retrieved


def chat_rag():
    """Interactive RAG chat loop."""
    print("Loading chapters...")
    chapters = load_all_chapters()
    
    print("Loading LLM...")
    model, tokenizer = load_model()
    
    print("\n" + "=" * 60)
    print("RAG System Ready 🚀")
    print("Type 'exit' to quit")
    print("=" * 60 + "\n")
    
    while True:
        query = input("You: ").strip()
        if query.lower() == "exit":
            break
        if not query:
            continue
        
        print("\n🔍 Retrieving relevant context...")
        response, sources = rag_generate(model, tokenizer, query, chapters)
        
        print(f"\n📚 LLM: {response}")
        print("\n📖 Sources used:")
        for src in sources:
            print(f"  - {src['chapter']} (score: {src['distance']:.2f})")
        print("-" * 50 + "\n")


if __name__ == "__main__":
    chat_rag()
