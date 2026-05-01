"""
vector_db.py — FAISS-backed dual-index vector store

Indexes:
  text_index  – IndexFlatL2  – SentenceTransformer (MiniLM) embeddings
  image_index – IndexFlatIP  – CLIP embeddings (cosine via inner-product on L2-normalised vecs)

Metadata is kept in parallel Python lists (text_meta / image_meta) and
persisted to disk alongside the FAISS indexes so that the store survives
server restarts.
"""

import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from core.config import settings

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
INDEX_DIR = settings.FAISS_INDEX_DIR
os.makedirs(INDEX_DIR, exist_ok=True)

TEXT_INDEX_PATH  = os.path.join(INDEX_DIR, "text_index.faiss")
IMAGE_INDEX_PATH = os.path.join(INDEX_DIR, "image_index.faiss")
TEXT_META_PATH   = os.path.join(INDEX_DIR, "text_meta.json")
IMAGE_META_PATH  = os.path.join(INDEX_DIR, "image_meta.json")

# ---------------------------------------------------------------------------
# Embedding dimensions
# ---------------------------------------------------------------------------
TEXT_DIM  = 384   # all-MiniLM-L6-v2
CLIP_DIM  = 512   # openai/clip-vit-base-patch32

# ---------------------------------------------------------------------------
# Lazy-loaded models (initialised only when first needed)
# ---------------------------------------------------------------------------
_text_embed_model = None

def _get_text_embed_model() -> SentenceTransformer:
    global _text_embed_model
    if _text_embed_model is None:
        print(f"[VectorDB] Loading text embedding model: {settings.TEXT_EMBED_MODEL}")
        _text_embed_model = SentenceTransformer(settings.TEXT_EMBED_MODEL)
    return _text_embed_model

# ---------------------------------------------------------------------------
# FAISS index helpers
# ---------------------------------------------------------------------------
def _load_or_create_text_index() -> faiss.IndexFlatL2:
    if os.path.exists(TEXT_INDEX_PATH):
        return faiss.read_index(TEXT_INDEX_PATH)
    return faiss.IndexFlatL2(TEXT_DIM)

def _load_or_create_image_index() -> faiss.IndexFlatIP:
    if os.path.exists(IMAGE_INDEX_PATH):
        return faiss.read_index(IMAGE_INDEX_PATH)
    return faiss.IndexFlatIP(CLIP_DIM)

def _load_meta(path: str) -> list:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def _save_meta(path: str, data: list):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def _persist_text(index, meta):
    faiss.write_index(index, TEXT_INDEX_PATH)
    _save_meta(TEXT_META_PATH, meta)

def _persist_image(index, meta):
    faiss.write_index(index, IMAGE_INDEX_PATH)
    _save_meta(IMAGE_META_PATH, meta)

# ---------------------------------------------------------------------------
# TEXT INDEX — add / search
# ---------------------------------------------------------------------------
def add_text_embeddings(chunks: list[str], source: str):
    """Embed chunks with MiniLM and add them to the text FAISS index."""
    if not chunks:
        return

    model  = _get_text_embed_model()
    index  = _load_or_create_text_index()
    meta   = _load_meta(TEXT_META_PATH)

    vectors = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=False)
    vectors = vectors.astype(np.float32)

    start_id = len(meta)
    index.add(vectors)

    for i, chunk in enumerate(chunks):
        meta.append({
            "id":      str(start_id + i),
            "content": chunk,
            "source":  source
        })

    _persist_text(index, meta)
    print(f"[VectorDB] Added {len(chunks)} text chunks from '{source}'. Total: {index.ntotal}")


def search_text(query: str, top_k: int = 10) -> list[dict]:
    """Return top-K text chunks most similar to query."""
    index = _load_or_create_text_index()
    meta  = _load_meta(TEXT_META_PATH)

    if index.ntotal == 0:
        return []

    model = _get_text_embed_model()
    q_vec = model.encode([query], convert_to_numpy=True).astype(np.float32)

    k      = min(top_k, index.ntotal)
    dists, ids = index.search(q_vec, k)

    results = []
    for dist, idx in zip(dists[0], ids[0]):
        if idx == -1:
            continue
        entry = meta[idx].copy()
        entry["score"] = float(dist)
        results.append(entry)
    return results


# ---------------------------------------------------------------------------
# IMAGE INDEX — add / search
# ---------------------------------------------------------------------------
def add_image_embeddings(embeddings: np.ndarray, image_paths: list[str], descriptions: list[str]):
    """Add pre-computed CLIP image embeddings to the image FAISS index."""
    if embeddings is None or len(embeddings) == 0:
        return

    index = _load_or_create_image_index()
    meta  = _load_meta(IMAGE_META_PATH)

    vecs = embeddings.astype(np.float32)
    faiss.normalize_L2(vecs)
    start_id = len(meta)
    index.add(vecs)

    for i, (path, desc) in enumerate(zip(image_paths, descriptions)):
        meta.append({
            "id":          str(start_id + i),
            "file_path":   path,
            "description": desc
        })

    _persist_image(index, meta)
    print(f"[VectorDB] Added {len(image_paths)} image embeddings. Total: {index.ntotal}")


def search_images(query_embedding: np.ndarray, top_k: int = 5) -> list[dict]:
    """Retrieve most similar images given a CLIP query embedding."""
    index = _load_or_create_image_index()
    meta  = _load_meta(IMAGE_META_PATH)

    if index.ntotal == 0:
        return []

    q_vec = query_embedding.reshape(1, -1).astype(np.float32)
    faiss.normalize_L2(q_vec)

    k      = min(top_k, index.ntotal)
    scores, ids = index.search(q_vec, k)

    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        entry = meta[idx].copy()
        entry["score"] = float(score)
        results.append(entry)
    return results


# ---------------------------------------------------------------------------
# Backward-compat shim
# ---------------------------------------------------------------------------
def upsert_chunks(chunks: list[str], filename: str):
    """Backward-compatible wrapper used by main.py upload endpoint."""
    add_text_embeddings(chunks, filename)
