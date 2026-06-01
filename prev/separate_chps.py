import os
import re
import pdfplumber
import faiss
import tiktoken
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from clean_text import clean_chapter_text

# ================= CONFIG =================
PDF_PATH = "book.pdf"
OUTPUT_DIR = "chapters_db"
CHUNK_SIZE = 400
OVERLAP = 60
EMBED_MODEL = "all-MiniLM-L6-v2"
NUM_KEYWORDS = 30
# =========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

tokenizer = tiktoken.get_encoding("cl100k_base")
embedder = SentenceTransformer(EMBED_MODEL)

CHAPTER_REGEX = re.compile(r"^Chapter\s+(\d+)", re.IGNORECASE)
STOP_REGEX = re.compile(r"^(Exercises|Problems|Review Questions)", re.IGNORECASE)
SKIP_REGEX = re.compile(r"^(Figure|Table)\s+\d+", re.IGNORECASE)

# -------- Helpers --------
def chunk_text(text):
    tokens = tokenizer.encode(text)
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = tokens[i:i + CHUNK_SIZE]
        chunks.append(tokenizer.decode(chunk))
        i += CHUNK_SIZE - OVERLAP
    return chunks

# Garbage patterns to exclude from keywords
GARBAGE_WORDS = {
    'fi', 'le', 'les', 'th', 'ned', 'defi', 'tion', 'tions', 'ing', 'ed', 'er',
    'ment', 'ments', 'ure', 'ures', 'ity', 'ies', 'ly', 'al', 'als', 'ble',
    'ness', 'ous', 'ive', 'ful', 'less', 'tion', 'sion', 'ary', 'ory', 'ism',
    'ist', 'ize', 'ise', 'ate', 'ent', 'ant', 'ance', 'ence', 'able', 'ible',
    'figure', 'table', 'chapter', 'page', 'example', 'section', 'see', 'also',
    'note', 'following', 'shown', 'using', 'used', 'use', 'uses', 'called',
    'like', 'new', 'different', 'number', 'numbers', 'value', 'values',
    'case', 'cases', 'type', 'types', 'way', 'ways', 'time', 'times',
    '00', '01', '10', '11', '32', '64', '16', '08',
}

def extract_keywords(text, k=NUM_KEYWORDS):
    """Extract meaningful keywords using TF-IDF with filtering."""
    try:
        # Use token pattern that requires at least 4 alphabetic characters
        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=k * 3,  # Get more candidates to filter from
            ngram_range=(1, 2),
            token_pattern=r'\b[a-zA-Z]{4,}\b',  # Only words with 4+ letters
            lowercase=True
        )
        vectorizer.fit([text])
        candidates = vectorizer.get_feature_names_out().tolist()
        
        # Filter out garbage words
        keywords = [
            kw for kw in candidates 
            if kw.lower() not in GARBAGE_WORDS 
            and not kw.isdigit()
            and len(kw) >= 4
        ]
        
        return keywords[:k]
    except:
        return []

def save_vector_db(chapter_id, chunks, keywords):
    if not chunks:
        print(f"⚠️ Skipping chapter{chapter_id} - no text chunks")
        return
    
    embeddings = embedder.encode(chunks, show_progress_bar=False)
    embeddings = np.array(embeddings).astype("float32")
    
    # Ensure 2D array (handle single chunk case)
    if len(embeddings.shape) == 1:
        embeddings = embeddings.reshape(1, -1)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    chapter_path = os.path.join(OUTPUT_DIR, f"chapter{chapter_id}")
    os.makedirs(chapter_path, exist_ok=True)

    faiss.write_index(index, os.path.join(chapter_path, "index.faiss"))

    with open(os.path.join(chapter_path, "texts.pkl"), "wb") as f:
        pickle.dump(chunks, f)

    with open(os.path.join(chapter_path, "meta.pkl"), "wb") as f:
        pickle.dump(
            {
                "chapter": chapter_id,
                "keywords": keywords
            },
            f
        )

    print(f"✅ Saved chapter{chapter_id} ({len(keywords)} keywords)")

# -------- Main --------
def process_pdf():
    current_chapter = None
    buffer = []

    with pdfplumber.open(PDF_PATH) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue

            for line in text.split("\n"):
                line = line.strip()
                if not line:
                    continue

                chap_match = CHAPTER_REGEX.match(line)
                if chap_match:
                    if current_chapter and buffer:
                        full_text = clean_chapter_text("\n".join(buffer))
                        chunks = chunk_text(full_text)
                        keywords = extract_keywords(full_text)
                        save_vector_db(current_chapter, chunks, keywords)
                        buffer = []

                    current_chapter = chap_match.group(1)
                    continue

                if STOP_REGEX.match(line):
                    continue

                if SKIP_REGEX.match(line):
                    continue

                if current_chapter:
                    buffer.append(line)

        # save last chapter
        if current_chapter and buffer:
            full_text = "\n".join(buffer)
            chunks = chunk_text(full_text)
            keywords = extract_keywords(full_text)
            save_vector_db(current_chapter, chunks, keywords)

# -------- Run --------
if __name__ == "__main__":
    process_pdf()
    print("Chapter-wise vector DBs with keywords created")
