from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from core.config import settings

# Initialize Neo4j graph connection
def get_graph():
    try:
        return Neo4jGraph(
            url=settings.NEO4J_URI,
            username=settings.NEO4J_USERNAME,
            password=settings.NEO4J_PASSWORD
        )
    except Exception as e:
        print(f"Failed to connect to Neo4j: {e}")
        return None

def extract_and_store_entities(chunks: list[str], filename: str):
    graph = get_graph()
    if not graph:
        return 0
    
    # Using LLM to extract entities and relationships
    llm = ChatGoogleGenerativeAI(temperature=0, google_api_key=settings.GEMINI_API_KEY, model="gemini-2.5-flash")
    llm_transformer = LLMGraphTransformer(llm=llm)
    
    # We create Document objects
    documents = [Document(page_content=chunk, metadata={"source": filename}) for chunk in chunks]
    
    try:
        graph_documents = llm_transformer.convert_to_graph_documents(documents)
        # Store to neo4j
        graph.add_graph_documents(
            graph_documents,
            baseEntityLabel=True,
            include_source=True
        )
        return len(graph_documents)
    except Exception as e:
        print(f"Graph extraction failed: {e}")
        return 0

def query_graph(query: str):
    graph = get_graph()
    if not graph:
        return []
    
    # We can use an LLM to generate Cypher query or do a basic keyword match in the graph.
    # For a robust approach, we can do GraphQA Chain, but for hybrid retrieval,
    # let's just use the GraphCypherQAChain or a simple direct query if we identify entities.
    from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
    
    llm = ChatGoogleGenerativeAI(temperature=0, google_api_key=settings.GEMINI_API_KEY, model="gemini-2.5-flash")
    chain = GraphCypherQAChain.from_llm(
        cypher_llm=llm,
        qa_llm=llm,
        graph=graph,
        verbose=True,
        return_direct=False
    )
    
    try:
        result = chain.invoke({"query": query})
        return result.get("result", "")
    except Exception as e:
        print(f"Graph query failed: {e}")
        return str(e)
