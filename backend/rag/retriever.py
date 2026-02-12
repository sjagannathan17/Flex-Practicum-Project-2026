"""
Vector retrieval from ChromaDB.
"""
from typing import Optional
from backend.core.database import get_collection, embed_text
from backend.core.config import TOP_K_RESULTS, SIMILARITY_THRESHOLD
from backend.core.cache import search_cache, cache_key

# Cache for query embeddings
_embedding_cache = {}


def _get_cached_embedding(query: str) -> list[float]:
    """Get cached embedding or compute new one."""
    key = cache_key(query)
    if key in _embedding_cache:
        return _embedding_cache[key]
    
    embedding = embed_text(query)
    _embedding_cache[key] = embedding
    
    # Limit cache size
    if len(_embedding_cache) > 100:
        # Remove oldest entries
        for old_key in list(_embedding_cache.keys())[:50]:
            _embedding_cache.pop(old_key, None)
    
    return embedding


def search_documents(
    query: str,
    company_filter: Optional[str] = None,
    filing_type_filter: Optional[str] = None,
    n_results: int = TOP_K_RESULTS,
) -> list[dict]:
    """
    Search ChromaDB for relevant document chunks.
    
    Args:
        query: Search query text
        company_filter: Optional company name to filter by
        filing_type_filter: Optional filing type to filter by
        n_results: Number of results to return
        
    Returns:
        List of document chunks with metadata and similarity scores
    """
    # Check cache first
    result_cache_key = f"search:{cache_key(query, company_filter, filing_type_filter, n_results)}"
    cached = search_cache.get(result_cache_key)
    if cached is not None:
        return cached
    
    collection = get_collection()
    
    # Get cached or compute embedding
    query_embedding = _get_cached_embedding(query)
    
    # Build where filter
    where_filter = None
    if company_filter or filing_type_filter:
        conditions = []
        if company_filter:
            conditions.append({"company": company_filter})
        if filing_type_filter:
            conditions.append({"filing_type": filing_type_filter})
        
        if len(conditions) == 1:
            where_filter = conditions[0]
        else:
            where_filter = {"$and": conditions}
    
    # Query ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where_filter,
        include=["documents", "metadatas", "distances"]
    )
    
    # Format results
    documents = []
    if results["documents"] and results["documents"][0]:
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            similarity = 1 - dist  # Convert distance to similarity
            
            # Skip low similarity results
            if similarity < SIMILARITY_THRESHOLD:
                continue
            
            documents.append({
                "content": doc,
                "metadata": meta,
                "similarity": round(similarity, 4),
                "source": meta.get("source_file", "Unknown"),
                "company": meta.get("company", "Unknown"),
                "filing_type": meta.get("filing_type", "Unknown"),
                "fiscal_year": meta.get("fiscal_year", "Unknown"),
                "quarter": meta.get("quarter", ""),
            })
    
    # Cache results
    search_cache.set(result_cache_key, documents)
    return documents


def search_by_company(company: str, n_results: int = 20) -> list[dict]:
    """Get recent documents for a specific company."""
    return search_documents(
        query=f"{company} capital expenditure investment strategy",
        company_filter=company,
        n_results=n_results,
    )


def search_cross_company(query: str, n_results: int = TOP_K_RESULTS) -> list[dict]:
    """Search across all companies for comparison."""
    return search_documents(query=query, n_results=n_results)


def get_company_documents(company: str, limit: int = 100) -> list[dict]:
    """Get all documents for a company (for analytics)."""
    # Check cache
    doc_cache_key = f"company_docs:{cache_key(company, limit)}"
    cached = search_cache.get(doc_cache_key)
    if cached is not None:
        return cached
    
    collection = get_collection()
    
    results = collection.get(
        where={"company": company},
        include=["documents", "metadatas"],
        limit=limit,
    )
    
    documents = []
    if results["documents"]:
        for doc, meta in zip(results["documents"], results["metadatas"]):
            documents.append({
                "content": doc,
                "metadata": meta,
            })
    
    # Cache results
    search_cache.set(doc_cache_key, documents)
    return documents
