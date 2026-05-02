"""
vector_db.py — Pinecone vector store

TWO separate Pinecone indexes:
  text_index  — dimension=384, metric=cosine  (SentenceTransformer MiniLM)
  image_index — dimension=512, metric=cosine  (CLIP openai/clip-vit-base-patch32)

Each index auto-creates on first use (ServerlessSpec, aws us-east-1).

Public API:
  add_text_embeddings(chunks, source)
  search_text(query, top_k)                    → list[dict]
  add_image_embeddings(embeddings, paths, desc)
  search_images(query_embedding, top_k)        → list[dict]
  upsert_chunks(chunks, filename)              ← backward-compat shim
"""

import uuid
import numpy as np
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from core.config import settings

# ---------------------------------------------------------------------------
# Embedding dimensions
# ---------------------------------------------------------------------------
TEXT_DIM = 384    # all-MiniLM-L6-v2
CLIP_DIM = 512    # openai/clip-vit-base-patch32

# ---------------------------------------------------------------------------
# Lazy singletons
# ---------------------------------------------------------------------------
_pc: Pinecone | None = None
_text_embed_model: SentenceTransformer | None = None


def _get_pinecone() -> Pinecone:
    global _pc
    if _pc is None:
        _pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    return _pc


def _get_text_embed_model() -> SentenceTransformer:
    global _text_embed_model
    if _text_embed_model is None:
        print(f"[VectorDB] Loading embedding model: {settings.TEXT_EMBED_MODEL}")
        _text_embed_model = SentenceTransformer(settings.TEXT_EMBED_MODEL)
    return _text_embed_model


# ---------------------------------------------------------------------------
# Index helpers — auto-create if absent
# ---------------------------------------------------------------------------
def _get_text_index():
    """Return (or create) the text Pinecone index."""
    pc         = _get_pinecone()
    index_name = settings.PINECONE_TEXT_INDEX
    existing   = [i.name for i in pc.list_indexes()]

    if index_name not in existing:
        print(f"[VectorDB] Creating Pinecone text index '{index_name}' (dim={TEXT_DIM})")
        pc.create_index(
            name      = index_name,
            dimension = TEXT_DIM,
            metric    = "cosine",
            spec      = ServerlessSpec(cloud="aws", region="us-east-1")
        )

    return pc.Index(index_name)


def _get_image_index():
    """Return (or create) the image Pinecone index."""
    pc         = _get_pinecone()
    index_name = settings.PINECONE_IMAGE_INDEX
    existing   = [i.name for i in pc.list_indexes()]

    if index_name not in existing:
        print(f"[VectorDB] Creating Pinecone image index '{index_name}' (dim={CLIP_DIM})")
        pc.create_index(
            name      = index_name,
            dimension = CLIP_DIM,
            metric    = "cosine",
            spec      = ServerlessSpec(cloud="aws", region="us-east-1")
        )

    return pc.Index(index_name)


# ---------------------------------------------------------------------------
# TEXT INDEX — add / search
# ---------------------------------------------------------------------------
def add_text_embeddings(chunks: list[str], source: str):
    """
    Embed chunks with MiniLM and upsert into the Pinecone text index.
    Metadata: { content, source }
    """
    if not chunks:
        return

    model  = _get_text_embed_model()
    index  = _get_text_index()

    vectors = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
    vectors = vectors.astype(np.float32)

    batch = []
    for i, (vec, chunk) in enumerate(zip(vectors, chunks)):
        # Pinecone metadata content limit ~40KB — truncate long chunks
        batch.append({
            "id":     str(uuid.uuid4()),
            "values": vec.tolist(),
            "metadata": {
                "content": chunk[:2000],   # truncate for metadata limit
                "source":  source,
                "id":      str(i)
            }
        })

    # Upsert in batches of 100 (Pinecone recommendation)
    for i in range(0, len(batch), 100):
        index.upsert(vectors=batch[i:i + 100])

    print(f"[VectorDB] Upserted {len(chunks)} text chunks from '{source}' → Pinecone '{settings.PINECONE_TEXT_INDEX}'")


def search_text(query: str, top_k: int = 10) -> list[dict]:
    """Query the text Pinecone index, returns list of dicts with content/source/score."""
    model  = _get_text_embed_model()
    index  = _get_text_index()

    q_vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0].tolist()

    result  = index.query(vector=q_vec, top_k=top_k, include_metadata=True)
    matches = result.get("matches", [])

    return [
        {
            "id":      m["id"],
            "content": m["metadata"].get("content", ""),
            "source":  m["metadata"].get("source", "unknown"),
            "score":   m["score"]
        }
        for m in matches
    ]


# ---------------------------------------------------------------------------
# IMAGE INDEX — add / search
# ---------------------------------------------------------------------------
def add_image_embeddings(embeddings: np.ndarray, image_paths: list[str], descriptions: list[str]):
    """
    Upsert pre-computed CLIP image embeddings into the Pinecone image index.
    embeddings: (N, 512) float32, already L2-normalised.
    """
    if embeddings is None or len(embeddings) == 0:
        return

    index = _get_image_index()
    vecs  = embeddings.astype(np.float32)

    # Normalise just in case
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs  = vecs / np.where(norms > 1e-9, norms, 1.0)

    batch = [
        {
            "id":     str(uuid.uuid4()),
            "values": vec.tolist(),
            "metadata": {
                "file_path":   path,
                "description": desc[:1000]
            }
        }
        for vec, path, desc in zip(vecs, image_paths, descriptions)
    ]

    for i in range(0, len(batch), 100):
        index.upsert(vectors=batch[i:i + 100])

    print(f"[VectorDB] Upserted {len(image_paths)} image embeddings → Pinecone '{settings.PINECONE_IMAGE_INDEX}'")


def search_images(query_embedding: np.ndarray, top_k: int = 5) -> list[dict]:
    """Query the image Pinecone index using a CLIP embedding."""
    index = _get_image_index()

    vec  = query_embedding.flatten().astype(np.float32)
    norm = np.linalg.norm(vec)
    vec  = (vec / norm).tolist() if norm > 1e-9 else vec.tolist()

    result  = index.query(vector=vec, top_k=top_k, include_metadata=True)
    matches = result.get("matches", [])

    return [
        {
            "id":          m["id"],
            "file_path":   m["metadata"].get("file_path", ""),
            "description": m["metadata"].get("description", ""),
            "score":       m["score"]
        }
        for m in matches
    ]


# ---------------------------------------------------------------------------
# Backward-compat shim
# ---------------------------------------------------------------------------
def upsert_chunks(chunks: list[str], filename: str):
    """Backward-compatible wrapper used by main.py upload endpoint."""
    add_text_embeddings(chunks, filename)
