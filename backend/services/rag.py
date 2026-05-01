from services.vector_db import query_vector_store
from services.graph_db import query_graph
from models.schemas import QueryResponse, SourceNode
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from core.config import settings

def hybrid_retrieval_and_answer(query: str) -> QueryResponse:
    # 1. Retrieve from Vector DB (Dense Retrieval)
    vector_results = query_vector_store(query, top_k=5)
    
    sources = []
    vector_context = ""
    for idx, (doc, score) in enumerate(vector_results):
        sources.append(SourceNode(
            id=str(idx),
            text=doc.page_content,
            metadata=doc.metadata
        ))
        vector_context += f"Chunk {idx}: {doc.page_content}\n"

    # 2. Retrieve from Graph DB
    graph_context = query_graph(query)
    
    # 3. Combine contexts and prompt LLM
    llm = ChatGoogleGenerativeAI(temperature=0, google_api_key=settings.GEMINI_API_KEY, model="gemini-2.5-flash")
    
    prompt = PromptTemplate(
        input_variables=["query", "vector_context", "graph_context"],
        template="""You are a helpful expert answering user questions based on provided data context.
        
        Vector Context (Semantic matches):
        {vector_context}
        
        Graph Context (Entity relationships):
        {graph_context}
        
        User Query: {query}
        
        Instructions:
        - Provide a clear, structured, and easy-to-read answer.
        - Start with a brief summary (maximum 2-3 sentences).
        - Follow with bullet points for key details, extracting only the most relevant information.
        - Keep the overall response concise and avoid overly complex explanations.
        - If the context does not contain the answer, say "I cannot answer this based on the provided documents.
        -Provide code  snippets if asked for"
        """
    )
    
    chain = prompt | llm
    
    answer = chain.invoke({
        "query": query,
        "vector_context": vector_context,
        "graph_context": str(graph_context)
    })
    
    return QueryResponse(
        answer=answer.content,
        sources=sources,
        graph_context=[{"graph_summary": str(graph_context)}] if graph_context else []
    )
