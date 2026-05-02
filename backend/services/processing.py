"""
processing.py — File ingestion and chunking
Supports: CSV, PDF, images (PNG/JPG/JPEG), plain text
"""

import os
import cv2
import pandas as pd
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter


def process_csv(filepath: str) -> list[str]:
    df      = pd.read_csv(filepath)
    records = df.to_dict(orient="records")
    texts   = [str(record) for record in records]
    return chunk_texts(texts)


def process_pdf(filepath: str) -> list[str]:
    doc  = fitz.open(filepath)
    text = ""
    for page in doc:
        text += page.get_text()
    return chunk_text(text)


def process_image(filepath: str) -> list[str]:
    """
    Preprocess image with OpenCV and store CLIP embedding in image Pinecone index.
    Returns a text description for the text index.
    """
    img = cv2.imread(filepath)
    if img is None:
        return [f"Image file could not be read: {filepath}"]

    # 1. Ask Ollama Vision model (llava) to describe the image
    from services.llm_service import describe_image
    print(f"[Processing] Generating description for {filepath} using Vision LLM...")
    llm_explanation = describe_image(filepath)

    height, width = img.shape[:2]
    
    description = (
        f"Image File: {os.path.basename(filepath)}\n"
        f"Resolution: {width}x{height}px\n\n"
        f"Explanation:\n{llm_explanation}"
    )

    # 2. CLIP embedding → image Pinecone index
    try:
        from services.clip_service import embed_image
        from services.vector_db   import add_image_embeddings

        embedding = embed_image(filepath)          # shape (512,)
        embedding = embedding.reshape(1, -1)       # shape (1, 512) — required by upsert
        add_image_embeddings(
            embeddings   = embedding,
            image_paths  = [filepath],
            descriptions = [description]
        )
        print(f"[Processing] CLIP embedding stored for: {filepath}")
    except Exception as e:
        print(f"[Processing] CLIP embedding failed for '{filepath}': {e}")

    # Return the description so it gets chunked into the text index too
    return [description], llm_explanation


def chunk_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_text(text)


def chunk_texts(texts: list[str]) -> list[str]:
    combined = "\n".join(texts)
    return chunk_text(combined)


def process_file(filepath: str, filename: str) -> list[str]:
    ext = filename.split(".")[-1].lower()
    if ext == "csv":
        return process_csv(filepath)
    elif ext == "pdf":
        return process_pdf(filepath)
    elif ext in ("png", "jpg", "jpeg"):
        return process_image(filepath)
    else:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return chunk_text(f.read())
