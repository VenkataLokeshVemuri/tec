"""
graph_db.py — Graph Database with Neo4j + offline JSON fallback

Strategy:
  1. Entity extraction   — regex-based (fully offline, no LLM needed)
  2. Graph storage       — Neo4j (Cypher) when available
  3. Graph query         — Cypher MATCH on entity nodes
  4. Fallback            — JSON co-occurrence graph when Neo4j is offline

Neo4j nodes  : (:Entity {name: str, source: str})
Neo4j edges  : (:Entity)-[:CO_OCCURS {sources: [str]}]->(:Entity)

Public API (same signatures as offline version):
  extract_and_store_entities(chunks, filename) → int
  query_graph(query)                           → str
"""

import os
import re
import json
from collections import defaultdict
from core.config import settings

# ---------------------------------------------------------------------------
# JSON fallback store (used when Neo4j is disabled or unreachable)
# ---------------------------------------------------------------------------
GRAPH_DIR  = settings.FAISS_INDEX_DIR
os.makedirs(GRAPH_DIR, exist_ok=True)
GRAPH_PATH = os.path.join(GRAPH_DIR, "graph_store.json")

_json_graph: dict = defaultdict(lambda: defaultdict(list))
_json_loaded = False


def _load_json_graph():
    global _json_graph, _json_loaded
    if _json_loaded:
        return
    if os.path.exists(GRAPH_PATH):
        with open(GRAPH_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        _json_graph = defaultdict(lambda: defaultdict(list))
        for entity, relations in raw.items():
            for rel_entity, sources in relations.items():
                _json_graph[entity][rel_entity] = sources
    _json_loaded = True


def _save_json_graph():
    with open(GRAPH_PATH, "w", encoding="utf-8") as f:
        json.dump({e: dict(rels) for e, rels in _json_graph.items()}, f, indent=2)


# ---------------------------------------------------------------------------
# Neo4j driver (lazy singleton)
# ---------------------------------------------------------------------------
_neo4j_driver = None
_neo4j_available = None   # None = not yet checked


def _get_neo4j_driver():
    """
    Returns the Neo4j driver if Neo4j is enabled and reachable.
    Returns None otherwise (triggers JSON fallback).
    """
    global _neo4j_driver, _neo4j_available

    if not settings.NEO4J_ENABLED:
        return None

    if _neo4j_available is False:
        return None

    if _neo4j_driver is not None:
        return _neo4j_driver

    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD)
        )
        # Verify connectivity
        driver.verify_connectivity()
        _neo4j_driver  = driver
        _neo4j_available = True
        print(f"[GraphDB] Connected to Neo4j at {settings.NEO4J_URI}")
        _ensure_constraints(driver)
        return driver
    except Exception as e:
        _neo4j_available = False
        print(f"[GraphDB] Neo4j unavailable ({e}). Using JSON fallback.")
        return None


def _ensure_constraints(driver):
    """Create a uniqueness constraint on Entity.name (runs once)."""
    try:
        with driver.session() as session:
            session.run(
                "CREATE CONSTRAINT entity_name_unique IF NOT EXISTS "
                "FOR (e:Entity) REQUIRE e.name IS UNIQUE"
            )
    except Exception:
        pass   # constraint may already exist or Neo4j version differs


# ---------------------------------------------------------------------------
# Entity extraction (offline, regex-based — no LLM required)
# ---------------------------------------------------------------------------
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "on",
    "at", "by", "for", "with", "from", "and", "or", "but", "not", "this",
    "that", "it", "its", "he", "she", "they", "we", "you", "i", "my",
    "your", "our", "their", "as", "if", "so", "then", "than", "when",
    "where", "which", "who", "what", "how", "any", "all", "each", "both",
    "error", "true", "false", "none", "null"
}


def _extract_entities(text: str) -> list[str]:
    """
    Extract capitalised noun phrases as candidate named entities.
    Returns deduplicated list (max 20 per chunk for performance).
    """
    tokens = re.findall(r"\b[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*\b", text)
    entities = []
    seen     = set()
    for tok in tokens:
        cleaned = tok.strip()
        if cleaned.lower() in _STOPWORDS or len(cleaned) < 2:
            continue
        if cleaned not in seen:
            seen.add(cleaned)
            entities.append(cleaned)
    return entities[:20]


# ---------------------------------------------------------------------------
# Neo4j operations
# ---------------------------------------------------------------------------
def _neo4j_store_entities(driver, entities: list[str], filename: str) -> int:
    """
    MERGE entity nodes and CO_OCCURS relationships into Neo4j.
    Returns number of relationships created.
    """
    count = 0
    with driver.session() as session:
        # Create/merge all entity nodes first
        for entity in entities:
            session.run(
                "MERGE (e:Entity {name: $name}) "
                "ON CREATE SET e.source = $source "
                "ON MATCH  SET e.source = e.source",
                name=entity, source=filename
            )

        # Create co-occurrence relationships between every pair
        for i, ent_a in enumerate(entities):
            for ent_b in entities[i + 1:]:
                result = session.run(
                    """
                    MATCH (a:Entity {name: $a}), (b:Entity {name: $b})
                    MERGE (a)-[r:CO_OCCURS]-(b)
                    ON CREATE SET r.sources = [$src]
                    ON MATCH  SET r.sources = CASE
                        WHEN $src IN r.sources THEN r.sources
                        ELSE r.sources + $src
                    END
                    RETURN r
                    """,
                    a=ent_a, b=ent_b, src=filename
                )
                if result.single():
                    count += 1

    return count


def _neo4j_query(driver, query_entities: list[str]) -> str:
    """
    Find matching entities in Neo4j and return their relationships as context.
    """
    lines = []
    with driver.session() as session:
        for qe in query_entities:
            # Try exact match first, then case-insensitive partial match
            result = session.run(
                """
                MATCH (a:Entity)-[r:CO_OCCURS]-(b:Entity)
                WHERE a.name = $name OR toLower(a.name) CONTAINS toLower($name)
                RETURN a.name AS entity, b.name AS related, r.sources AS sources
                LIMIT 5
                """,
                name=qe
            )
            for record in result:
                sources = ", ".join(record["sources"]) if record["sources"] else "unknown"
                lines.append(
                    f"[Graph] '{record['entity']}' co-occurs with "
                    f"'{record['related']}' (sources: {sources})"
                )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON fallback operations
# ---------------------------------------------------------------------------
def _json_store_entities(entities: list[str], filename: str) -> int:
    _load_json_graph()
    count = 0
    for i, ent_a in enumerate(entities):
        for ent_b in entities[i + 1:]:
            if filename not in _json_graph[ent_a][ent_b]:
                _json_graph[ent_a][ent_b].append(filename)
                _json_graph[ent_b][ent_a].append(filename)
                count += 1
    _save_json_graph()
    return count


def _json_query(query_entities: list[str]) -> str:
    _load_json_graph()
    lines = []
    for qe in query_entities:
        if qe in _json_graph:
            for rel_ent, sources in list(_json_graph[qe].items())[:5]:
                lines.append(
                    f"[Graph] '{qe}' co-occurs with '{rel_ent}' "
                    f"(sources: {', '.join(set(sources))})"
                )
        else:
            for node in _json_graph:
                if qe.lower() in node.lower() or node.lower() in qe.lower():
                    for rel_ent, sources in list(_json_graph[node].items())[:3]:
                        lines.append(
                            f"[Graph] '{node}' co-occurs with '{rel_ent}' "
                            f"(sources: {', '.join(set(sources))})"
                        )
                    break
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def extract_and_store_entities(chunks: list[str], filename: str) -> int:
    """
    Extract named entities from chunks and store in Neo4j (or JSON fallback).
    Returns total entity count stored.
    """
    driver = _get_neo4j_driver()
    total  = 0

    for chunk in chunks:
        entities = _extract_entities(chunk)
        if not entities:
            continue

        if driver:
            total += _neo4j_store_entities(driver, entities, filename)
        else:
            total += _json_store_entities(entities, filename)

    backend = "Neo4j" if driver else "JSON fallback"
    print(f"[GraphDB] Stored entities from '{filename}' → {backend} (relationships: {total})")
    return total


def query_graph(query: str) -> str:
    """
    Find entities from the query in the graph and return relationship context.
    Uses Neo4j if available, JSON fallback otherwise.
    """
    query_entities = _extract_entities(query)
    if not query_entities:
        return ""

    driver = _get_neo4j_driver()

    if driver:
        return _neo4j_query(driver, query_entities)
    else:
        return _json_query(query_entities)


def get_graph_backend() -> str:
    """Returns which graph backend is currently active (for health/status endpoint)."""
    driver = _get_neo4j_driver()
    return "neo4j" if driver else "json_fallback"
