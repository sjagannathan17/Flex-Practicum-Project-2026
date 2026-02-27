"""
Chat API routes with SSE streaming, query analysis, and smart routing.
"""
import re
import json
import uuid
from typing import Optional

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.rag.retriever import search_documents, search_cross_company
from backend.rag.generator import generate_response_streaming, SYSTEM_PROMPT
from backend.rag.assembled_retriever import get_assembled_retriever
from backend.rag.memory import (
    add_message,
    get_conversation_history,
    clear_session,
    get_session_info,
    get_all_sessions,
    cleanup_expired_sessions,
)
from backend.rag.web_search import search_web, format_web_results_for_context
from backend.core.config import COMPANIES, COMPANY_NAME_TO_TICKER

router = APIRouter()

COMPANY_NAMES = {
    config["name"].split()[0].lower(): config["name"].split()[0]
    for config in COMPANIES.values()
}
COMPANY_NAMES.update({t.lower(): config["name"].split()[0] for t, config in COMPANIES.items()})

METRIC_KEYWORDS = {
    "capex": ["capex", "capital expenditure", "capital spending", "pp&e",
              "property plant and equipment", "property, plant"],
    "revenue": ["revenue", "sales", "top line", "top-line"],
    "margin": ["margin", "gross margin", "operating margin", "profit margin"],
    "ai": ["ai", "artificial intelligence", "machine learning", "gpu",
           "data center", "datacenter", "hyperscale"],
    "guidance": ["guidance", "outlook", "forecast", "expect"],
}

COMPARISON_TRIGGERS = [
    "compare", "comparison", "vs", "versus", "against", "between",
    "how does", "how do", "relative to", "compared to",
    "all companies", "each company", "every company",
]


class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    mode: str = "rag"  # "rag" | "web" | "hybrid" | "assembled" (NEW)
    include_web: bool = False
    company_filter: Optional[str] = None
    retrieval_strategy: str = "auto"  # For assembled mode: "auto" | "vector" | "bm25" | "hybrid" | "table"
    use_reranking: bool = True


def _detect_companies(query: str) -> list[str]:
    """Detect company names or tickers in the query."""
    q_lower = query.lower()
    found = []
    for key, canonical in COMPANY_NAMES.items():
        if key in q_lower and canonical not in found:
            found.append(canonical)
    return found


def _detect_metrics(query: str) -> list[str]:
    """Detect financial metric categories in the query."""
    q_lower = query.lower()
    found = []
    for metric, keywords in METRIC_KEYWORDS.items():
        if any(kw in q_lower for kw in keywords):
            found.append(metric)
    return found


def _detect_year(query: str) -> Optional[str]:
    """Detect fiscal year references."""
    match = re.search(r'\b(20[1-3]\d)\b', query)
    if match:
        return match.group(1)
    fy = re.search(r'\bFY\s*(\d{2,4})\b', query, re.IGNORECASE)
    if fy:
        y = fy.group(1)
        return f"20{y}" if len(y) == 2 else y
    return None


def _is_comparison_query(query: str, companies: list[str]) -> bool:
    """Check if the query is asking for a comparison."""
    q_lower = query.lower()
    if len(companies) >= 2:
        return True
    return any(trigger in q_lower for trigger in COMPARISON_TRIGGERS)


def _build_context(docs: list[dict]) -> str:
    """Format retrieved documents into a context string for the LLM."""
    if not docs:
        return ""
    parts = []
    for i, doc in enumerate(docs, 1):
        header = f"[{doc['company']} | {doc['filing_type']} | {doc['fiscal_year']}"
        if doc.get("quarter"):
            header += f" {doc['quarter']}"
        header += f" | sim={doc['similarity']:.2f}]"
        # Use full parent content when available (page/section with tables included)
        # This ensures financial tables (like cash flow statements) are fully visible
        content = doc.get("parent_content") or doc["content"]
        if len(content) > 3000:
            content = content[:3000] + "..."
        parts.append(f"{header}\n{content}")
    return "\n\n---\n\n".join(parts)


def _sse_event(event: str, data: dict) -> str:
    """Format a server-sent event."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _stream_response(request: ChatRequest):
    """Main streaming generator that handles the full RAG pipeline."""
    session_id = request.session_id or str(uuid.uuid4())
    query = request.query.strip()

    # Step 1: Analyse the query
    companies = _detect_companies(query)
    metrics = _detect_metrics(query)
    year = _detect_year(query)
    is_comparison = _is_comparison_query(query, companies)

    yield _sse_event("step", {
        "icon": "🔍",
        "label": "Query Analysis",
        "detail": f"Companies: {companies or 'auto'} | Metrics: {metrics or 'general'} | Year: {year or 'any'}",
    })

    # Step 2: Determine routing
    use_agentic = is_comparison and len(companies) >= 2

    if use_agentic:
        yield _sse_event("step", {
            "icon": "🤖",
            "label": "Routing",
            "detail": "Using agentic multi-step retrieval for comparison query",
        })
    else:
        yield _sse_event("step", {
            "icon": "📄",
            "label": "Routing",
            "detail": "Using single-call retrieval",
        })

    # Step 3: Retrieve documents
    if use_agentic:
        try:
            from backend.rag.agentic import agentic_stream
            add_message(session_id, "user", query)
            async for event_type, event_data in agentic_stream(query):
                yield _sse_event(event_type, event_data)
            return
        except ImportError:
            pass

    # NEW: Use AssembledRetriever for "assembled" mode
    docs = []
    assembled_context = ""
    query_analysis = None
    
    if request.mode == "assembled":
        retriever = get_assembled_retriever()
        result = retriever.search(
            query=query,
            company=request.company_filter or (companies[0] if len(companies) == 1 else None),
            top_k=15,
            strategy=request.retrieval_strategy,
            use_parent_expansion=True,
            use_reranking=request.use_reranking,
        )
        docs = [
            {
                "company": d.get("company", ""),
                "filing_type": d.get("filing_type", ""),
                "fiscal_year": d.get("fiscal_year", ""),
                "quarter": d.get("quarter", ""),
                "similarity": d.get("score", 0),
                "content": d.get("parent_content") or d.get("content", ""),
                "source": d.get("source", ""),
                "page_num": d.get("page_num", 0),
                "section_header": d.get("section_header", ""),
            }
            for d in result["results"]
        ]
        assembled_context = result.get("context", "")
        query_analysis = result.get("analysis", {})
        
        yield _sse_event("step", {
            "icon": "🔧",
            "label": "Assembled Retriever",
            "detail": f"Strategy: {result.get('strategy_used', 'auto')} | Type: {query_analysis.get('query_type', 'unknown')}",
        })
    elif companies and len(companies) == 1:
        docs = search_documents(query, company_filter=companies[0], n_results=15, use_reranking=request.use_reranking)
    elif is_comparison:
        docs = search_cross_company(query, n_results=30)
    else:
        docs = search_documents(query, n_results=15, use_reranking=request.use_reranking)

    yield _sse_event("step", {
        "icon": "📚",
        "label": "Retrieved",
        "detail": f"{len(docs)} document chunks",
    })

    # Step 4: Optional web search
    web_context = ""
    if request.include_web:
        try:
            web_query = query
            if companies:
                web_query = f"{' '.join(companies)} {query}"
            web_results = await search_web(web_query)
            if web_results:
                web_context = format_web_results_for_context(web_results)
                yield _sse_event("step", {
                    "icon": "🌐",
                    "label": "Web Search",
                    "detail": f"{len(web_results)} web results found",
                })
        except Exception:
            pass

    # Step 5: Build context and generate
    # For assembled mode, use pre-built context; otherwise build from docs
    if request.mode == "assembled" and assembled_context:
        context = assembled_context
    else:
        context = _build_context(docs)

    if not context and not web_context:
        yield _sse_event("token", {"text": "I couldn't find relevant documents to answer your question. Try rephrasing or check that the data has been ingested."})
        yield _sse_event("done", {"session_id": session_id})
        return

    # Save user message to session
    add_message(session_id, "user", query)

    yield _sse_event("step", {
        "icon": "✨",
        "label": "Generating",
        "detail": "Streaming response from Claude",
    })

    # Stream tokens
    full_response = ""
    for chunk in generate_response_streaming(query, context, web_context):
        full_response += chunk
        yield _sse_event("token", {"text": chunk})

    # Save assistant response to session
    add_message(session_id, "assistant", full_response)

    yield _sse_event("done", {"session_id": session_id})


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """SSE streaming chat endpoint."""
    return StreamingResponse(
        _stream_response(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/chat")
async def chat(request: ChatRequest):
    """Non-streaming chat endpoint (returns full response)."""
    from backend.rag.generator import generate_response

    session_id = request.session_id or str(uuid.uuid4())
    query = request.query.strip()

    companies = _detect_companies(query)
    query_analysis = None
    strategy_used = None

    # NEW: Use AssembledRetriever for "assembled" mode
    if request.mode == "assembled":
        retriever = get_assembled_retriever()
        result = retriever.search(
            query=query,
            company=request.company_filter or (companies[0] if len(companies) == 1 else None),
            top_k=15,
            strategy=request.retrieval_strategy,
            use_parent_expansion=True,
            use_reranking=request.use_reranking,
        )
        docs = [
            {
                "company": d.get("company", ""),
                "filing_type": d.get("filing_type", ""),
                "fiscal_year": d.get("fiscal_year", ""),
                "source": d.get("source", ""),
                "similarity": d.get("score", 0),
                "content": d.get("parent_content") or d.get("content", ""),
                "page_num": d.get("page_num", 0),
                "section_header": d.get("section_header", ""),
            }
            for d in result["results"]
        ]
        context = result.get("context", "")
        query_analysis = result.get("analysis", {})
        strategy_used = result.get("strategy_used", "auto")
    else:
        if companies and len(companies) == 1:
            docs = search_documents(query, company_filter=companies[0], n_results=15, use_reranking=request.use_reranking)
        else:
            docs = search_documents(query, n_results=15, use_reranking=request.use_reranking)
        context = _build_context(docs)

    web_context = ""
    if request.include_web:
        try:
            web_results = await search_web(query)
            if web_results:
                web_context = format_web_results_for_context(web_results)
        except Exception:
            pass

    add_message(session_id, "user", query)
    response_text = generate_response(query, context, web_context)
    add_message(session_id, "assistant", response_text)

    response_dict = {
        "response": response_text,
        "session_id": session_id,
        "sources": [
            {
                "company": d.get("company", ""),
                "source": d.get("source", ""),
                "fiscal_year": d.get("fiscal_year", ""),
                "page_num": d.get("page_num"),
                "section_header": d.get("section_header"),
            }
            for d in docs[:5]
        ],
    }
    
    # Include assembled mode metadata
    if query_analysis:
        response_dict["query_analysis"] = query_analysis
        response_dict["strategy_used"] = strategy_used
    
    return response_dict


# ---------------------------------------------------------------------------
# Session management endpoints
# ---------------------------------------------------------------------------

@router.get("/chat/sessions")
async def list_sessions():
    """List all active chat sessions."""
    cleanup_expired_sessions()
    return {"sessions": get_all_sessions()}


@router.get("/chat/sessions/{session_id}")
async def get_session(session_id: str):
    """Get info about a specific session."""
    return get_session_info(session_id)


@router.get("/chat/sessions/{session_id}/history")
async def get_history(session_id: str):
    """Get conversation history for a session."""
    return {"messages": get_conversation_history(session_id)}


@router.delete("/chat/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a chat session."""
    clear_session(session_id)
    return {"status": "deleted", "session_id": session_id}
