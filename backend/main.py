"""
main.py — Multi-Modal Graph RAG API (Fully Offline)

No API keys required. All processing runs locally:
  - Pinecone for vector storage
  - Ollama (Phi-3 Mini) for LLM
  - CLIP for multimodal embeddings
  - CrossEncoder for reranking
  - Offline co-occurrence graph
"""

import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from models.schemas      import QueryRequest, QueryResponse, FileUploadResponse
from services.processing import process_file
from services.vector_db  import upsert_chunks
from services.graph_db   import extract_and_store_entities
from services.rag        import hybrid_retrieval_and_answer

app = FastAPI(title="Multi-Modal Graph RAG API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/api/upload", response_model=FileUploadResponse)
async def upload_file(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 1. Process & chunk
        ext = file.filename.rsplit(".", 1)[-1].lower()
        image_description = None

        if ext in ("png", "jpg", "jpeg"):
            # process_image returns (chunks, llm_explanation)
            result = process_file(file_path, file.filename)
            if isinstance(result, tuple):
                chunks, image_description = result
            else:
                chunks = result
        else:
            chunks = process_file(file_path, file.filename)

        if not chunks:
            raise HTTPException(status_code=400, detail="Could not extract text from file")

        # 2. Upsert to Pinecone text index
        upsert_chunks(chunks, file.filename)

        # 3. Graph entity extraction
        entities_count = extract_and_store_entities(chunks, file.filename)

        return FileUploadResponse(
            filename=file.filename,
            status="Success",
            chunks_processed=len(chunks),
            entities_extracted=entities_count,
            image_description=image_description,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query", response_model=QueryResponse)
async def query_data(request: QueryRequest):
    try:
        response = hybrid_retrieval_and_answer(request.query)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health():
    from services.graph_db import get_graph_backend
    return {
        "status":       "ok",
        "version":      "2.0",
        "mode":         "offline",
        "graph_backend": get_graph_backend()   # "neo4j" or "json_fallback"
    }

@app.get("/api/debug_env")
async def debug_env():
    from core.config import settings
    return {"PINECONE_API_KEY": settings.PINECONE_API_KEY}


@app.post("/api/clear-index")
async def clear_index():
    """Delete all vectors from Pinecone text + image indexes so stale data can be flushed."""
    from services.vector_db import _get_pinecone
    from core.config import settings
    pc = _get_pinecone()
    results = {}
    for index_name in [settings.PINECONE_TEXT_INDEX, settings.PINECONE_IMAGE_INDEX]:
        existing = [i.name for i in pc.list_indexes()]
        if index_name in existing:
            pc.Index(index_name).delete(delete_all=True)
            results[index_name] = "cleared"
        else:
            results[index_name] = "not found"
    return {"status": "ok", "indexes": results}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
