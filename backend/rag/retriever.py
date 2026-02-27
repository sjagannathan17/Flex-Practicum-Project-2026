"""
Vector retrieval module for RAG pipeline.
Handles document search with year detection, recency boosting, query expansion,
BM25 keyword search, and Reciprocal Rank Fusion (RRF).
"""
import re
from typing import Optional

from backend.core.database import get_collection, embed_text


# ---------------------------------------------------------------------------
# QUERY EXPANSION — catches different CapEx terminology across companies
# ---------------------------------------------------------------------------
_FINANCIAL_SYNONYMS = {
    "capex": [
        "capital expenditures",
        "purchases of property and equipment",
        "acquisition of property, plant and equipment",
        "purchase of property, plant and equipment",
        "additions to property and equipment",
        "capital spending",
        "payments for property and equipment",
        "property, plant and equipment additions",
    ],
    "revenue": [
        "net revenue", "net revenues", "total revenue", "sales",
    ],
    "profit": [
        "net income", "operating income", "gross profit", "earnings",
    ],
    "free cash flow": [
        "FCF", "cash flow from operations minus capex",
    ],
}


def _expand_query(query: str) -> str:
    """Expand query with financial synonyms for better recall."""
    q_lower = query.lower()
    expansions = []
    for term, synonyms in _FINANCIAL_SYNONYMS.items():
        if term in q_lower:
            expansions.extend(synonyms[:3])
    if expansions:
        return query + " " + " ".join(expansions)
    return query


# ---------------------------------------------------------------------------
# BM25 INDEX — keyword-based search for exact term matching
# ---------------------------------------------------------------------------
_bm25_index = None
_bm25_corpus = None
_bm25_doc_ids = None


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer."""
    return re.findall(r'\b\w+\b', text.lower())


def _get_bm25_index():
    """Build or return cached BM25 index from ChromaDB documents."""
    global _bm25_index, _bm25_corpus, _bm25_doc_ids
    if _bm25_index is not None:
        return _bm25_index, _bm25_corpus, _bm25_doc_ids

    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        return None, None, None

    collection = get_collection()
    count = collection.count()
    if count == 0:
        return None, None, None

    batch_size = 5000
    all_docs = []
    all_ids = []
    all_metas = []

    for offset in range(0, count, batch_size):
        results = collection.get(
            include=["documents", "metadatas"],
            limit=batch_size,
            offset=offset,
        )
        all_docs.extend(results.get("documents", []))
        all_ids.extend(results.get("ids", []))
        all_metas.extend(results.get("metadatas", []))

    tokenized = [_tokenize(doc) for doc in all_docs]
    _bm25_index = BM25Okapi(tokenized)
    _bm25_corpus = list(zip(all_docs, all_metas, all_ids))
    _bm25_doc_ids = all_ids
    return _bm25_index, _bm25_corpus, _bm25_doc_ids


def _bm25_search(
    query: str,
    company_filter: Optional[str] = None,
    n_results: int = 20,
) -> list[dict]:
    """Search using BM25 keyword matching."""
    bm25, corpus, doc_ids = _get_bm25_index()
    if bm25 is None or not corpus:
        return []

    tokens = _tokenize(query)
    scores = bm25.get_scores(tokens)

    scored = list(zip(scores, corpus))
    if company_filter:
        scored = [(s, (doc, meta, did)) for s, (doc, meta, did) in scored
                  if meta.get("company") == company_filter]

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:n_results]

    max_score = top[0][0] if top and top[0][0] > 0 else 1.0
    results = []
    for score, (doc, meta, did) in top:
        if score <= 0:
            continue
        results.append({
            "content": doc,
            "company": meta.get("company", "Unknown"),
            "source": meta.get("source_file", "Unknown"),
            "filing_type": meta.get("filing_type", "Unknown"),
            "fiscal_year": meta.get("fiscal_year", "Unknown"),
            "quarter": meta.get("quarter", ""),
            "similarity": round(score / max_score, 4),
        })
    return results


def _reciprocal_rank_fusion(
    vector_results: list[dict],
    bm25_results: list[dict],
    k: int = 60,
    n_results: int = 20,
) -> list[dict]:
    """Merge vector and BM25 results using Reciprocal Rank Fusion.
    RRF_score = sum(1 / (k + rank)) across both result lists."""
    doc_scores: dict[str, float] = {}
    doc_map: dict[str, dict] = {}

    for rank, doc in enumerate(vector_results):
        key = f"{doc.get('company', '')}_{doc.get('source', '')}_{doc['content'][:80]}"
        doc_scores[key] = doc_scores.get(key, 0) + 1.0 / (k + rank + 1)
        doc_map[key] = doc

    for rank, doc in enumerate(bm25_results):
        key = f"{doc.get('company', '')}_{doc.get('source', '')}_{doc['content'][:80]}"
        doc_scores[key] = doc_scores.get(key, 0) + 1.0 / (k + rank + 1)
        if key not in doc_map:
            doc_map[key] = doc

    sorted_keys = sorted(doc_scores, key=doc_scores.get, reverse=True)
    merged = []
    for key in sorted_keys[:n_results]:
        doc = doc_map[key]
        doc["similarity"] = round(doc_scores[key], 4)
        merged.append(doc)
    return merged


def _extract_year_from_query(query: str) -> Optional[str]:
    """Extract a 4-digit year from the query string."""
    match = re.search(r'\b(20[1-3]\d)\b', query)
    if match:
        return match.group(1)

    fy_match = re.search(r'\bFY\s*(\d{2,4})\b', query, re.IGNORECASE)
    if fy_match:
        y = fy_match.group(1)
        if len(y) == 2:
            return f"20{y}"
        return y

    return None


def _fiscal_year_variants(year: str) -> list[str]:
    """Generate all common fiscal-year string variants for a given year."""
    short = year[-2:]
    return [
        year,
        f"FY{year}",
        f"FY{short}",
        f"fiscal {year}",
        f"fiscal year {year}",
        f"Fiscal Year {year}",
        f"FY {year}",
        f"FY {short}",
    ]


def _should_auto_detect_year(query: str) -> bool:
    """Determine whether the query contains an explicit year reference."""
    if re.search(r'\b(20[1-3]\d)\b', query):
        return True
    if re.search(r'\bFY\s*\d{2,4}\b', query, re.IGNORECASE):
        return True
    if re.search(r'\bfiscal\s+(year\s+)?\d{4}\b', query, re.IGNORECASE):
        return True
    return False


_RECENCY_KEYWORDS = {
    "latest", "recent", "newest", "most recent", "current", "last quarter",
    "this year", "this quarter", "updated", "new",
}


def _wants_latest(query: str) -> bool:
    """Check if the query implies the user wants the most recent data."""
    q_lower = query.lower()
    for kw in _RECENCY_KEYWORDS:
        if kw in q_lower:
            return True
    if re.search(r'\brecent\b', q_lower):
        return True
    return False


def _fy_sort_key(doc: dict) -> str:
    """Sort key that orders documents by fiscal year descending."""
    fy = doc.get("fiscal_year", "") or ""
    match = re.search(r'(\d{4})', str(fy))
    return match.group(1) if match else "0000"


def _quarter_sort_key(doc: dict) -> int:
    """Sort key that orders documents by quarter descending."""
    q = doc.get("quarter", "") or ""
    match = re.search(r'Q(\d)', str(q))
    return int(match.group(1)) if match else 0


def search_documents(
    query: str,
    company_filter: Optional[str] = None,
    filing_type_filter: Optional[str] = None,
    n_results: int = 20,
) -> list[dict]:
    """
    Search ChromaDB for relevant document chunks with re-ranking.

    Applies year boosting when a year is detected in the query and
    recency boosting when the query implies the user wants recent data.
    """
    collection = get_collection()
    if collection.count() == 0:
        return []

    expanded_query = _expand_query(query)
    query_embedding = embed_text(expanded_query)

    where_filter = None
    if company_filter and filing_type_filter:
        where_filter = {
            "$and": [
                {"company": company_filter},
                {"filing_type": filing_type_filter},
            ]
        }
    elif company_filter:
        where_filter = {"company": company_filter}
    elif filing_type_filter:
        where_filter = {"filing_type": filing_type_filter}

    fetch_n = min(n_results * 3, collection.count())

    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=fetch_n,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )
    except Exception:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=fetch_n,
            include=["documents", "metadatas", "distances"],
        )

    if not results or not results["documents"] or not results["documents"][0]:
        return []

    docs = []
    for doc_text, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        similarity = 1 - distance
        docs.append({
            "content": doc_text,
            "company": metadata.get("company", "Unknown"),
            "source": metadata.get("source_file", metadata.get("source", "Unknown")),
            "filing_type": metadata.get("filing_type", "Unknown"),
            "fiscal_year": metadata.get("fiscal_year", "Unknown"),
            "quarter": metadata.get("quarter", ""),
            "similarity": round(similarity, 4),
        })

    docs.sort(key=lambda d: d["similarity"], reverse=True)

    # Hybrid: merge with BM25 results using RRF
    bm25_results = _bm25_search(query, company_filter=company_filter, n_results=n_results * 2)
    if bm25_results:
        docs = _reciprocal_rank_fusion(docs, bm25_results, n_results=n_results * 2)
    else:
        docs = docs[:n_results * 2]

    # Apply boosts AFTER RRF merge so they aren't overwritten

    # Year boosting
    detected_year = _extract_year_from_query(query)
    if detected_year:
        variants = _fiscal_year_variants(detected_year)
        for doc in docs:
            fy = str(doc.get("fiscal_year", ""))
            if any(v.lower() in fy.lower() for v in variants):
                doc["similarity"] += 0.15

    # Recency filtering — when user asks for recent/latest, drop old documents
    if _wants_latest(query) and not detected_year:
        recent_fy = set()
        all_fys = sorted(set(_fy_sort_key(d) for d in docs if _fy_sort_key(d) != "0000"), reverse=True)
        recent_fy = set(all_fys[:3])  # Keep top 3 fiscal years
        if recent_fy:
            recent_docs = [d for d in docs if _fy_sort_key(d) in recent_fy]
            if len(recent_docs) >= 5:
                docs = recent_docs

    docs.sort(key=lambda d: d["similarity"], reverse=True)
    return docs[:n_results]


def search_by_company(
    query: str,
    company: str,
    n_results: int = 20,
) -> list[dict]:
    """Convenience wrapper to search within a single company."""
    return search_documents(query, company_filter=company, n_results=n_results)


def search_cross_company(
    query: str,
    n_results: int = 50,
) -> list[dict]:
    """Search across all companies without a company filter."""
    return search_documents(query, n_results=n_results)


def get_company_documents(company: str, limit: int = 100) -> list[dict]:
    """
    Retrieve raw document chunks for a company (no query embedding needed).

    Returns dicts with 'content' and 'metadata' keys, matching the format
    expected by analytics modules.
    """
    collection = get_collection()
    if collection.count() == 0:
        return []

    try:
        results = collection.get(
            where={"company": company},
            include=["documents", "metadatas"],
            limit=limit,
        )
    except Exception:
        return []

    docs = []
    for doc_text, metadata in zip(
        results.get("documents", []),
        results.get("metadatas", []),
    ):
        docs.append({
            "content": doc_text,
            "metadata": metadata,
        })

    return docs
