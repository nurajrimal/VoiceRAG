"""
rag_pipeline.py
Manages the full RAG pipeline:
  - Create embeddings
  - Store/load FAISS vector database
  - Retrieve relevant chunks
  - Generate answers via Ollama LLM
"""

from dotenv import load_dotenv
import os
load_dotenv()
LLM_MODEL = os.getenv("LLM_MODEL", "mistral")
import pickle
import requests
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss


# ─── Configuration ────────────────────────────────────────────────────────────

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"   # Fast, free, runs locally
VECTOR_DB_PATH       = "vector_db/faiss.index"
CHUNKS_PATH          = "vector_db/chunks.pkl"
OLLAMA_URL           = "http://localhost:11434/api/generate"
DEFAULT_LLM_MODEL    = "mistral"             # Change to llama3 or gemma if preferred
TOP_K                = 5                     # Number of chunks to retrieve


# ─── Embedding ─────────────────────────────────────────────────────────────────

_embedding_model = None

def get_embedding_model() -> SentenceTransformer:
    """Load embedding model once and cache it."""
    global _embedding_model
    if _embedding_model is None:
        print(f"[RAG] Loading embedding model: {EMBEDDING_MODEL_NAME}")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def embed_texts(texts: list[str]) -> np.ndarray:
    """Convert a list of text chunks into embedding vectors."""
    model = get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings.astype("float32")


# ─── Vector Store ──────────────────────────────────────────────────────────────

def build_vector_store(chunks: list[str]) -> faiss.IndexFlatL2:
    """
    Build a FAISS index from document chunks and persist to disk.

    Args:
        chunks: List of text chunks from the PDF

    Returns:
        FAISS index
    """
    print("[RAG] Building vector store...")
    embeddings = embed_texts(chunks)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Persist index and chunks
    os.makedirs("vector_db", exist_ok=True)
    faiss.write_index(index, VECTOR_DB_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print(f"[RAG] Vector store built with {index.ntotal} vectors.")
    return index


def load_vector_store():
    """Load a previously saved FAISS index and chunks from disk."""
    if not os.path.exists(VECTOR_DB_PATH) or not os.path.exists(CHUNKS_PATH):
        return None, None

    index = faiss.read_index(VECTOR_DB_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)

    print(f"[RAG] Loaded vector store with {index.ntotal} vectors.")
    return index, chunks


def vector_store_exists() -> bool:
    return os.path.exists(VECTOR_DB_PATH) and os.path.exists(CHUNKS_PATH)


# ─── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve_relevant_chunks(
    query: str,
    index: faiss.IndexFlatL2,
    chunks: list[str],
    top_k: int = TOP_K
) -> list[str]:
    """
    Retrieve the top-k most relevant chunks for a query.

    Args:
        query: User's question
        index: FAISS index
        chunks: Original text chunks
        top_k: Number of results to retrieve

    Returns:
        List of relevant text chunks
    """
    model = get_embedding_model()
    query_embedding = model.encode([query], convert_to_numpy=True).astype("float32")

    distances, indices = index.search(query_embedding, top_k)
    relevant_chunks = [chunks[i] for i in indices[0] if i < len(chunks)]

    print(f"[RAG] Retrieved {len(relevant_chunks)} chunks for query.")
    return relevant_chunks


# ─── LLM Answer Generation ─────────────────────────────────────────────────────

def build_prompt(query: str, context_chunks: list[str]) -> str:
    """Build a RAG prompt from the query and retrieved context."""
    context = "\n\n---\n\n".join(context_chunks)
    prompt = f"""You are a helpful assistant that answers questions based strictly on the provided document context.

CONTEXT FROM DOCUMENT:
{context}

QUESTION:
{query}

INSTRUCTIONS:
- Answer only using the context above.
- If the answer is not in the context, say "I couldn't find that information in the document."
- Be concise and accurate.
- Do not make up information.

ANSWER:"""
    return prompt


def generate_answer_ollama(
    query: str,
    context_chunks: list[str],
    model: str = DEFAULT_LLM_MODEL
) -> str:
    """
    Generate an answer using Ollama running locally.

    Args:
        query: User's question
        context_chunks: Retrieved relevant chunks
        model: Ollama model name (mistral, llama3, gemma, etc.)

    Returns:
        Generated answer as string
    """
    prompt = build_prompt(query, context_chunks)

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,   # Low temperature for factual answers
            "top_p": 0.9,
            "num_predict": 512
        }
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        answer = result.get("response", "").strip()
        return answer if answer else "No answer generated."
    except requests.exceptions.ConnectionError:
        return (
            "❌ Ollama is not running. Please start it with:\n"
            "  ollama serve\n"
            "Then pull a model with:\n"
            "  ollama pull mistral"
        )
    except requests.exceptions.Timeout:
        return "❌ LLM response timed out. Try a smaller model or shorter document."
    except Exception as e:
        return f"❌ Error generating answer: {str(e)}"


def check_ollama_status(model: str = DEFAULT_LLM_MODEL) -> tuple[bool, str]:
    """Check if Ollama is running and the model is available."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = [m["name"] for m in response.json().get("models", [])]
            model_available = any(model in m for m in models)
            if model_available:
                return True, f"✅ Ollama running. Model '{model}' is ready."
            else:
                return False, (
                    f"⚠️ Ollama running but model '{model}' not found.\n"
                    f"Available: {', '.join(models) or 'none'}\n"
                    f"Run: ollama pull {model}"
                )
    except Exception:
        pass
    return False, (
        "❌ Ollama not running.\n"
        "Install: https://ollama.ai\n"
        "Start: ollama serve\n"
        "Pull model: ollama pull mistral"
    )


# ─── Full RAG Query Pipeline ────────────────────────────────────────────────────

def answer_question(
    query: str,
    index: faiss.IndexFlatL2,
    chunks: list[str],
    model: str = DEFAULT_LLM_MODEL,
    top_k: int = TOP_K
) -> tuple[str, list[str]]:
    """
    Full RAG pipeline: query → retrieve → generate → return.

    Returns:
        (answer, relevant_chunks)
    """
    relevant_chunks = retrieve_relevant_chunks(query, index, chunks, top_k)
    answer = generate_answer_ollama(query, relevant_chunks, model)
    return answer, relevant_chunks