import os
from dotenv import load_dotenv

load_dotenv(override=True)

class Settings:
    # Ollama (local LLM) settings
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "gemma3:1b")
    OLLAMA_VISION_MODEL: str = os.getenv("OLLAMA_VISION_MODEL", "llava")

    # Text embedding model (SentenceTransformer MiniLM)
    TEXT_EMBED_MODEL: str = os.getenv("TEXT_EMBED_MODEL", "all-MiniLM-L6-v2")

    # CLIP model for multimodal embeddings
    CLIP_MODEL: str = os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch32")

    # Reranker cross-encoder model
    RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Local persistence directory (for graph JSON fallback)
    LOCAL_STORE_DIR: str = os.getenv("LOCAL_STORE_DIR", "local_store")

    # Retrieval hyper-params
    RETRIEVE_TOP_K: int = int(os.getenv("RETRIEVE_TOP_K", "10"))
    RERANK_TOP_N: int = int(os.getenv("RERANK_TOP_N", "3"))

    # Pinecone vector database
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_TEXT_INDEX: str = os.getenv("PINECONE_TEXT_INDEX", "graph-rag-text")
    PINECONE_IMAGE_INDEX: str = os.getenv("PINECONE_IMAGE_INDEX", "graph-rag-images")

    # Neo4j graph database (optional — falls back to JSON graph if unavailable)
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USERNAME: str = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "madhukar")
    NEO4J_ENABLED: bool = os.getenv("NEO4J_ENABLED", "true").lower() == "true"

settings = Settings()
