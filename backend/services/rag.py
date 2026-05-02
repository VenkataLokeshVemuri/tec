"""
rag.py — Multi-Modal Graph RAG Pipeline (fully offline)

Pipeline:
  Query
    ↓ search_text(top_k=10)      [Pinecone + MiniLM]
    ↓ search_images(top_k=5)     [Pinecone + CLIP]
    ↓ rerank(top_n=3)            [CrossEncoder]
    ↓ query_graph()              [offline co-occurrence graph]
    ↓ fuse context
    ↓ ask_llm()                  [Ollama Phi-3 Mini]
    ↓ QueryResponse { answer, text_sources, image_sources }
"""

from services.vector_db   import search_text, search_images
from services.reranker    import rerank
from services.llm_service import ask_llm
from services.graph_db    import query_graph
from services.clip_service import embed_text_clip
from models.schemas       import QueryResponse, SourceNode
from core.config          import settings


def hybrid_retrieval_and_answer(query: str) -> QueryResponse:
    # 1. Dense text retrieval
    raw_text_results = search_text(query, top_k=settings.RETRIEVE_TOP_K)

    # 2. Multimodal image retrieval (CLIP text→image)
    query_clip_vec    = embed_text_clip(query)
    raw_image_results = search_images(query_clip_vec, top_k=5)

    # 3. Rerank text results → top 3
    reranked_text = rerank(
        query     = query,
        documents = raw_text_results,
        top_n     = settings.RERANK_TOP_N
    )

    # 4. Graph context
    graph_ctx = query_graph(query)

    # 5. Fuse context for LLM
    context_parts = []

    for i, doc in enumerate(reranked_text, 1):
        src   = doc.get("source", "unknown")
        score = doc.get("rerank_score", 0.0)
        context_parts.append(
            f"[Text {i} | source: {src} | relevance: {score:.3f}]\n{doc['content']}"
        )

    if graph_ctx:
        context_parts.append(f"[Graph Relationships]\n{graph_ctx}")

    if raw_image_results:
        img_lines = [
            f"[Image {i+1}] {img.get('description', img.get('file_path', ''))} "
            f"(score: {img.get('score', 0):.3f})"
            for i, img in enumerate(raw_image_results[:3])
        ]
        context_parts.append("[Relevant Images]\n" + "\n".join(img_lines))

    fused_context = "\n\n".join(context_parts)

    # 6. LLM answer via Ollama
    answer = ask_llm(context=fused_context, query=query)

    # 7. Build response
    text_sources = [
        SourceNode(
            id       = doc.get("id", str(i)),
            text     = doc.get("content", ""),
            metadata = {
                "source":       doc.get("source", "unknown"),
                "rerank_score": doc.get("rerank_score", 0.0)
            }
        )
        for i, doc in enumerate(reranked_text)
    ]

    image_sources = [
        SourceNode(
            id       = img.get("id", str(i)),
            text     = img.get("description", ""),
            metadata = {
                "file_path": img.get("file_path", ""),
                "score":     img.get("score", 0.0)
            }
        )
        for i, img in enumerate(raw_image_results[:3])
    ]

    return QueryResponse(
        answer        = answer,
        sources       = text_sources,
        text_sources  = text_sources,
        image_sources = image_sources,
        graph_context = [{"graph_summary": graph_ctx}] if graph_ctx else []
    )
