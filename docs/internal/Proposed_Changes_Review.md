# Proposed Changes Review — Brutal Validation

**Reviewer:** Sricharan  
**Date:** February 23, 2026  
**Source:** CHANGELOG.md (friend's proposed modifications inspired by RAG-Challenge-2)

---

## Changes to SKIP (Not Recommended)

### 1. LLM Provider Migration: Anthropic → OpenAI (Change #21)

**Proposal:** Replace all Anthropic/Claude API calls with OpenAI/GPT-4o across generator.py, sentiment.py, agentic.py, ai_extractor.py, and config.py.

**Why this is a problem:**

- Our entire working codebase uses Anthropic Claude. The .env file has a working ANTHROPIC_API_KEY. Switching providers means everyone needs a new API key.
- Our SETUP.md and team email tell teammates to sign up at console.anthropic.com — this would make all of that wrong.
- The generator prompt we carefully tuned (YTD vs quarterly, negative numbers, all 8 CapEx labels, unit headers) was written for Claude's format. It would need retesting with GPT-4o.
- No stated reason for switching. Claude has a 200K context window vs GPT-4o's 128K, which matters when processing long SEC filings.
- This change touches 5 files and would break the app until everyone gets new API keys.

**Verdict:** Gratuitous provider swap with no justification. Breaks our existing working setup.

---

### 2. Independent Company Collections — One ChromaDB Collection Per Company (Change #7)

**Proposal:** Instead of one ChromaDB collection with company metadata filtering, create 5 separate collections (flex_documents, jabil_documents, etc.).

**Why this is a problem:**

- Our current setup already isolates company results using `where={"company": "Flex"}` metadata filtering — this does the exact same thing without any architectural change.
- Splitting into 5 collections means rewriting database.py, retriever.py, build_chromadb.py, and every analytics module that calls get_collection().
- Cross-company queries (e.g., "Compare CapEx across all 5 companies") now need to search 5 separate collections and merge results, adding complexity.
- 5x the ChromaDB management overhead for zero measurable improvement.

**Verdict:** Massive complexity for no benefit. Metadata filtering already solves this.

---

### 3. Comparison Query Pipeline — N Separate API Calls (Change #9)

**Proposal:** For comparison queries like "Compare CapEx across all 5 companies", make 5 separate retrieval calls + 5 separate LLM calls + 1 merge call = 11 API calls total.

**Why this is a problem:**

- Our current system handles comparisons in 1 retrieval + 1 LLM call = 2 calls. Claude is perfectly capable of comparing data from multiple companies in a single prompt.
- 5x the latency — the user waits for 5 sequential answers instead of 1.
- 5x the API cost per comparison query.
- More complex code with no proven accuracy benefit.

**Verdict:** Over-engineered. Adds cost and latency for no demonstrated improvement.

---

### 4. Event Extraction System (Change #11)

**Proposal:** Create a full event extraction framework with CapExEvent schema, BucketL1/L2 categories, EvidenceSpan, two-stage extraction (rule-based screening + LLM extraction), and event deduplication.

**Why this is a problem:**

- Nothing in the app actually uses it. No UI page displays events. No API endpoint serves them. No pipeline feeds data into it.
- It is a standalone module that sits in the codebase doing nothing.
- Adds complexity and code to maintain with no current use case.

**Verdict:** Dead code. Should only be built when there is an actual feature that needs it.

---

### 5. Hardcoded "Intelligence" Pages (Changes #14, #15, #16)

**Proposal:** Create three new frontend pages — AI Investments Tracker (Big 5 hyperscalers), News Monitor, and Competitor Investments. Data served by a new intelligence.py API module.

**Why this is a problem:**

- All data is entirely hardcoded as static Python dictionaries in intelligence.py. Microsoft's 2026 capex, Meta's growth percentages, Stargate project details — all manually typed.
- There is no scraping, no API feed, no database, no automated updates. The moment these numbers change, someone has to manually edit the Python file.
- Presents static data as if it were a live intelligence system, which is misleading.

**Verdict:** Demo pages for a presentation, not real functionality. Acceptable only if clearly labeled as mockups.

---

### 6. CLI Pipeline (Change #12)

**Proposal:** Create scripts/pipeline_cli.py with commands like `download-filings`, `parse-documents`, `build-index`, `run-query`, `evaluate`.

**Why this is a problem:**

- The app already has build_chromadb.py for building the index, API endpoints for querying (/api/chat), and API endpoints for ingestion (/api/ingestion/check-filings).
- This duplicates existing functionality behind a different interface.
- Nobody on the team has requested a CLI tool.

**Verdict:** Duplicates existing functionality. Low priority.

---

### 7. LLM Reranking with gpt-4o-mini (Change #1.3)

**Proposal:** After retrieving ~60 candidate documents, send them to gpt-4o-mini for scoring on a 0-10 scale before passing top results to the main LLM.

**Why this is a problem:**

- Adds an OpenAI dependency (conflicts with our Anthropic setup).
- Every query now makes 2 LLM calls instead of 1, roughly doubling API cost per query (~$0.032 vs current ~$0.015).
- Our retriever already does year boosting, recency boosting, company filtering, and similarity ranking — which handles most of the same problem without an extra API call.
- The cost adds up: at 100 queries/day, this is an extra ~$50/month just for reranking.

**Verdict:** Potentially useful but too expensive for our budget. Our existing boosting and filtering handles most of the same problem for free.

---

## Changes Worth Keeping

For reference, the following proposed changes are sound and worth integrating:

| Change | Why it's good |
|--------|---------------|
| Hybrid Vector + BM25 Retrieval (#1.2) | Combines semantic search with exact keyword matching. Proven improvement for financial terms. |
| Query Expansion (#2) | Catches different CapEx terminology across companies. Simple to implement. |
| Table Serialization (#4) | Makes financial tables searchable by embedding them as linearized text. |
| Structure-Aware Chunking (#5) | Better than fixed word-count chunks. Keeps tables intact, respects section boundaries. |
| Rich Metadata (#6) | page_num, section_header, table_type enable powerful filtering. |
| Chain of Thought (#8) | Forces step-by-step reasoning for numeric extraction. Improves accuracy. |
| Chart Description Component (#17) | Simple, meets client requirement that charts have descriptions. |
| Dependencies Fix (#24) | Replacing strict pins with minimum versions fixes install failures. |

---

*Note: The "good" changes listed above would need to be adapted to work with our Anthropic/Claude setup rather than OpenAI, since we are not switching providers.*
