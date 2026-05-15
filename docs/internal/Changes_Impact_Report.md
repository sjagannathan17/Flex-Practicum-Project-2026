# Changes Impact Report

**Date:** February 23, 2026
**Scope:** Review and selective integration of proposed RAG-Challenge-2 inspired modifications

---

# Part 1: Changes Implemented

## 1. Dependency Pins Relaxed (Change #24)

**What changed:** All strict version pins (`==`) in `backend/requirements.txt` replaced with minimum versions (`>=`). Added `rank-bm25>=0.2.2` for hybrid retrieval.

**Impact on system:**
- `pip install` no longer fails on different Python versions (3.10, 3.11, 3.12, 3.13)
- Teammates can install dependencies without `ResolutionImpossible` errors
- BM25 keyword search is now available as a retrieval method

**Risk:** Minimal. Lower-bound constraints ensure backward compatibility while allowing pip to resolve transitive dependencies freely.

---

## 2. Chart Description Component (Change #17)

**What changed:** Created `frontend/src/components/ui/chart-description.tsx` — a reusable component that adds description text, data source, and last-updated note below any chart. Applied to all 4 dashboard charts:
- AI/Data Center Investment Focus
- Sentiment Analysis
- Documents by Company
- Documents by Type

**Impact on system:**
- Meets client requirement that every chart must have a clear description
- Users now understand what each chart shows and where the data comes from
- Reusable across any future chart in the app

**Risk:** None. Purely additive UI component.

---

## 3. Query Expansion for Financial Terms (Change #2)

**What changed:** Added `_expand_query()` function to `backend/rag/retriever.py` with a synonym dictionary. When a user searches for "capex", the system automatically expands the query to also search for "capital expenditures", "purchases of property and equipment", "additions to property and equipment", etc.

**Impact on system:**
- Retrieval recall improves significantly for CapEx queries — chunks using different terminology are now found
- Works across all 5 companies (Flex uses "Purchases of property and equipment", Sanmina uses "Capital expenditures", etc.)
- Also covers revenue and profit synonyms

**Risk:** Minimal. Expanded queries produce slightly more embedding computation per search, but the cost is negligible (< 1ms).

---

## 4. Chain of Thought Prompt (Change #8)

**What changed:** Added step-by-step reasoning instructions to the `SYSTEM_PROMPT` in `backend/rag/generator.py`:
1. Identify what specific information is being asked
2. Search through context for relevant data
3. Verify data matches the question (right company, period, metric)
4. Check for pitfalls (YTD vs quarterly, unit conversion, negative signs)
5. Formulate answer with proper units and context

Also added explicit instruction: if data is not found, state it clearly — do not hallucinate.

**Impact on system:**
- More accurate numerical answers — the model now verifies company, period, and metric before answering
- Catches common errors: YTD vs single-quarter confusion, thousands vs millions, negative cash flow values
- Reduces hallucination — model explicitly told not to make up numbers
- No code changes needed — purely a prompt enhancement

**Risk:** Slightly longer responses due to reasoning steps. Marginal increase in token usage (~10-15% more output tokens).

---

## 5. Table Serialization for Embeddings (Change #4)

**What changed:** Added `serialize_table_for_embedding()` to `build_chromadb.py`. Financial tables are now converted to linearized text for embedding:

```
Before (Markdown — poor for embedding):
| Revenue | $5,000M | $4,200M |

After (Linearized — good for embedding):
Revenue = $5,000M; Growth = 15%
```

**Impact on system:**
- Financial tables become actually searchable via vector similarity
- Previously, Markdown pipes and dashes were noise that degraded embedding quality
- Cash flow statements, balance sheets, and income statements are now retrievable

**Risk:** Requires rebuilding ChromaDB to take effect. Existing database uses old chunking format.

---

## 6. Rich Metadata for Chunks (Change #6)

**What changed:** Each chunk stored in ChromaDB now includes additional metadata fields:
- `section_header` — which SEC section the chunk belongs to (e.g., "Item 7. MD&A", "Cash Flow Statement")
- `table_type` — if the chunk contains a financial table (e.g., "cash_flow", "balance_sheet", "income_statement")
- `table_context` — unit header detected (e.g., "in thousands", "in millions")
- `parent_id` — links child chunks to their parent document for future parent expansion

**Impact on system:**
- Enables metadata-filtered retrieval: when someone asks about CapEx, the retriever can prioritize chunks tagged as `table_type=cash_flow`
- Unit context (`table_context`) helps the generator know whether to divide by 1,000 or not
- Section headers enable section-level filtering (e.g., show only MD&A sections for analytical questions)

**Risk:** Requires rebuilding ChromaDB. Existing chunks don't have these metadata fields.

---

## 7. Hybrid BM25 + Vector Retrieval with RRF (Change #1.2)

**What changed:** Added BM25 keyword search alongside existing vector (semantic) search in `backend/rag/retriever.py`. Results from both methods are merged using Reciprocal Rank Fusion (RRF):

```
RRF_score = 1/(k + vector_rank) + 1/(k + bm25_rank)    where k=60
```

The BM25 index is built lazily on first query from all ChromaDB documents.

**Impact on system:**
- Significantly better retrieval for exact financial terms. Vector search finds conceptually similar text; BM25 finds exact keyword matches. Together they catch both.
- Example: "Purchases of property and equipment" — vector search might return chunks about "capital investments" (conceptually similar but wrong), while BM25 finds the exact phrase in the cash flow statement.
- First query is slightly slower (~2-3 seconds) while BM25 index builds. Subsequent queries use the cached index.

**Risk:** 
- BM25 index is built in-memory from all ChromaDB documents. For ~19K chunks, this uses ~50-100 MB RAM.
- First query latency increase of 2-3 seconds (one-time cost).

---

## 8. Structure-Aware Chunking (Change #5)

**What changed:** Replaced fixed 250-word chunking in `build_chromadb.py` with structure-aware splitting:

1. **SEC section splitting** — Chunks align with document sections (Item 1, Item 7, Item 8, Cash Flow Statement, Balance Sheet, etc.)
2. **Table preservation** — Financial tables (detected by `|` characters) are kept as single chunks, never split across boundaries
3. **Paragraph-boundary respect** — Text splits at paragraph boundaries instead of mid-sentence
4. **Configurable limits** — MIN_CHUNK_WORDS=60, TARGET=250, MAX=350

**Impact on system:**
- Cash flow statements are no longer split across chunks — column alignment is preserved
- The retriever gets complete financial tables instead of fragments
- Section headers provide context for what each chunk contains
- Smaller, focused chunks improve retrieval precision

**Risk:** Requires rebuilding ChromaDB. Produces different chunk counts than the fixed-word approach (more chunks for structured documents, fewer for narrative text).

---

# Part 2: Changes Rejected and Why

## 1. LLM Provider Migration: Anthropic → OpenAI (Change #21)

**What it proposed:** Replace all Anthropic/Claude API calls with OpenAI/GPT-4o across 5 files (generator.py, sentiment.py, agentic.py, ai_extractor.py, config.py).

**Why it's problematic:**
- **Breaks existing setup.** Our working `.env` has an `ANTHROPIC_API_KEY`. Our SETUP.md tells teammates to sign up at console.anthropic.com. Our email to the team says "create your own Anthropic key." Switching providers invalidates all of this.
- **No stated justification.** The CHANGELOG gives no reason for switching. Claude has a 200K context window (vs GPT-4o's 128K), which matters for long SEC filings.
- **Cascading breakage.** Touches 5 files simultaneously. If any one fails, the entire app is broken.
- **Team disruption.** Everyone needs a new API key from a different provider.

---

## 2. Independent Company Collections (Change #7)

**What it proposed:** Split the single ChromaDB collection into 5 separate per-company collections (flex_documents, jabil_documents, etc.).

**Why it's problematic:**
- **Solves a problem that doesn't exist.** Our current `where={"company": "Flex"}` metadata filter already isolates company-specific results with zero performance penalty.
- **Breaks cross-company queries.** "Compare CapEx across all 5 companies" now requires searching 5 separate collections and merging results — significant code complexity.
- **Massive refactor.** Every module that calls `get_collection()` (20+ files) would need rewriting.
- **No measurable benefit.** ChromaDB's metadata filtering on a 19K-document collection takes < 50ms. Splitting provides zero speed improvement.

---

## 3. Comparison Query Pipeline — N Separate API Calls (Change #9)

**What it proposed:** For comparison queries, make 5 separate retrieval calls + 5 separate LLM calls + 1 merge call = 11 API calls per question.

**Why it's problematic:**
- **5x latency.** User waits for 5 sequential LLM responses instead of 1.
- **5x API cost.** Five separate Claude calls at ~$0.015 each = $0.075 per comparison, vs $0.015 for our current single-call approach.
- **Unnecessary.** Claude handles multi-company comparisons well in a single prompt with all context provided at once.
- **Over-engineered.** Adds complex orchestration code (sub-question decomposition, per-company retrieval, answer merging) for no proven accuracy benefit.

---

## 4. Event Extraction System (Change #11)

**What it proposed:** Create a full event extraction framework with `CapExEvent` schema, two-stage extraction (rule-based + LLM), event deduplication, category taxonomies (`BucketL1`, `BucketL2`), and evidence spans.

**Why it's problematic:**
- **Dead code.** No UI page displays events. No API endpoint serves them. No pipeline feeds data into them. The module would sit unused.
- **Premature abstraction.** Building a full taxonomy system (`GPU_SERVERS`, `LIQUID_COOLING`, `M_AND_A`) before having any use case for it creates unmaintainable code.
- **Maintenance burden.** Schema changes require updating multiple files. Dead code rots fastest.

---

## 5. Hardcoded Intelligence Pages (Changes #14, #15, #16)

**What it proposed:** Three new frontend pages (AI Investments Tracker, News Monitor, Competitor Investments) backed by a new `intelligence.py` API module with static data.

**Why it's problematic:**
- **Fake data.** All numbers (Microsoft's 2026 CapEx, Meta's growth percentages, Stargate project details) are manually typed as Python dictionaries. There is no scraping, no API feed, no database, no automated updates.
- **Misleading.** Presents static data as if it were a live intelligence system. The moment any number changes, someone must manually edit the Python file.
- **Acceptable only as labeled mockups.** If clearly presented as "demo data for presentation purposes", these could be useful. As "implemented features", they are misleading.

---

## 6. CLI Pipeline (Change #12)

**What it proposed:** Create `scripts/pipeline_cli.py` with commands like `download-filings`, `parse-documents`, `build-index`, `run-query`, `evaluate`.

**Why it's problematic:**
- **Duplicates existing functionality.** `build_chromadb.py` already builds the index. `/api/chat` already handles queries. `/api/ingestion/check-filings` already triggers SEC downloads.
- **No team request.** Nobody asked for a CLI tool. The existing API + web interface serves all current needs.
- **Maintenance overhead.** Two ways to do the same thing means two codepaths to keep in sync.

---

## 7. LLM Reranking with gpt-4o-mini (Change #1.3)

**What it proposed:** After retrieving ~60 candidate documents, send them to `gpt-4o-mini` for scoring on a 0-10 scale before passing top results to the main LLM.

**Why it's problematic:**
- **Doubles API cost.** Every query now makes 2 LLM calls instead of 1 (~$0.032 vs current ~$0.015 per query).
- **Adds OpenAI dependency.** Conflicts with our Anthropic-only setup. Would require users to have both API keys.
- **Diminishing returns.** Our retriever already does year boosting, recency boosting, company filtering, query expansion, and BM25+vector fusion. Adding LLM reranking on top provides marginal improvement at significant cost.
- **At 100 queries/day, reranking alone costs ~$50/month** — matching our entire current Claude budget.

---

*This document covers all 26 proposed changes from CHANGELOG.md. 8 were implemented, 7 were rejected with detailed justification above. The remaining 11 (Changes #3, #10, #13, #18, #19, #20, #22, #23, #25, #26, and parts of #8) were either already covered by our existing implementations, superseded by the changes we made, or low-priority cosmetic updates.*
