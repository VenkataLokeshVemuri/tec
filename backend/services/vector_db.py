from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from core.config import settings
import os

pc = Pinecone(api_key=settings.PINECONE_API_KEY)
embeddings_model = GoogleGenerativeAIEmbeddings(google_api_key=settings.GEMINI_API_KEY, model="models/gemini-embedding-2")

def init_pinecone():
    index_name = settings.PINECONE_INDEX_NAME
    if index_name not in pc.list_indexes().names():
        try:
            pc.create_index(
                name=index_name,
                dimension=3072, # Gemini embeddings dimension
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1" # Update depending on the pinecone account setup
                )
            )
        except Exception as e:
            print(f"Index creation failed or index already exists: {e}")

def get_vector_store():
    init_pinecone()
    index = pc.Index(settings.PINECONE_INDEX_NAME)
    return PineconeVectorStore(index=index, embedding=embeddings_model, text_key="text")

def upsert_chunks(chunks: list[str], filename: str):
    vector_store = get_vector_store()
    metadatas = [{"source": filename} for _ in chunks]
    vector_store.add_texts(texts=chunks, metadatas=metadatas)

def query_vector_store(query: str, top_k: int = 5):
    vector_store = get_vector_store()
    results = vector_store.similarity_search_with_score(query, k=top_k)
    return results
