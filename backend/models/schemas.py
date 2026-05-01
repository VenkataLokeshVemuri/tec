from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class QueryRequest(BaseModel):
    query: str


class SourceNode(BaseModel):
    id:       str
    text:     str
    metadata: Dict[str, Any]


class QueryResponse(BaseModel):
    answer:        str
    sources:       List[SourceNode]           # backward-compat
    text_sources:  List[SourceNode]
    image_sources: List[SourceNode]
    graph_context: Optional[List[Dict[str, Any]]] = None


class FileUploadResponse(BaseModel):
    filename:           str
    status:             str
    chunks_processed:   int
    entities_extracted: int
