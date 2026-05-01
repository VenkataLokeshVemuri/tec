import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from models.schemas import QueryRequest, QueryResponse, FileUploadResponse
from services.processing import process_file
from services.vector_db import upsert_chunks
from services.graph_db import extract_and_store_entities
from services.rag import hybrid_retrieval_and_answer

app = FastAPI(title="Multi-Modal Graph RAG API", version="1.0")

# CORS middleware for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
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
            
        # 1. Process and Chunk
        chunks = process_file(file_path, file.filename)
        if not chunks:
            raise HTTPException(status_code=400, detail="Could not extract text from file")
            
        # 2. Upsert to Vector DB (Pinecone)
        upsert_chunks(chunks, file.filename)
        
        # 3. Graph RAG (Neo4j)
        entities_count = extract_and_store_entities(chunks, file.filename)
        
        return FileUploadResponse(
            filename=file.filename,
            status="Success",
            chunks_processed=len(chunks),
            entities_extracted=entities_count
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query", response_model=QueryResponse)
async def query_data(request: QueryRequest):
    try:
        response = hybrid_retrieval_and_answer(request.query)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
