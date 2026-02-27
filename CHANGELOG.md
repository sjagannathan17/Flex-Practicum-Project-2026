# Project Modifications Changelog

## Summary

This document outlines all modifications made to the Flex Competitive Intelligence Platform after cloning from the repository. Changes are primarily inspired by [RAG-Challenge-2](https://github.com/ilyaRice/RAG-Challenge-2), the winning solution for Enterprise RAG Challenge 2.

---

# Part A: RAG System Optimizations (Inspired by RAG-Challenge-2)

## 1. Assembled Retriever System

### File Created:
- `backend/rag/assembled_retriever.py`

### Features Implemented:

#### 1.1 Pluggable Retrieval Strategies
```python
class BaseRetriever (Abstract)
├── VectorRetriever    # ChromaDB semantic search
├── BM25Retriever      # Keyword search (rank_bm25)
└── TableRetriever     # Specialized table chunk search
```

**Impact:** Enables flexible combination of retrieval methods instead of hardcoded single approach.

#### 1.2 Hybrid Vector + BM25 Retrieval
- Implements Reciprocal Rank Fusion (RRF) for merging results
- Combines semantic understanding (vector) with exact keyword matching (BM25)
- Formula: `RRF_score = Σ 1/(k + rank)` where k=60

**Impact:** More robust recall, especially for financial terms that need exact matching.

#### 1.3 LLM Reranking
```python
class LLMReranker:
    SYSTEM_PROMPT = """Score based on "Can this passage DIRECTLY SUPPORT answering?"
    NOT just "Does it mention the topic?"
    
    Scoring criteria (0-10):
    - 10: Contains EXACT answer
    - 8-9: Data that DIRECTLY supports computing answer
    - 6-7: Closely related data
    - 4-5: Mentions topic but lacks specific data
    - 2-3: Only tangentially related
    - 0-1: Not useful
    
    CapEx synonyms: "Purchases of property and equipment", 
    "Capital expenditures", "Additions to PP&E"...
    """
```

- Uses weighted scoring: 0.3 vector score + 0.7 LLM score
- Model: `gpt-4o-mini` for cost efficiency

**Impact:** Dramatically improves retrieval precision by focusing on answer-supporting content, not just topic relevance.

#### 1.4 Query Routing
```python
class QueryRouter:
    def analyze(query) -> QueryAnalysis:
        # Detects:
        - query_type: numeric | comparison | descriptive | table_lookup | summary
        - companies: list of mentioned companies
        - is_comparison: bool
        - recommended_top_k: int
        - use_reranking: bool
        - parent_granularity: "page" | "section"
    
    def get_route_type(query) -> RouteType:
        # R1_COMPARISON: Cross-company comparison
        # R2_REALTIME: Recency-focused queries
        # R3_PRECISION: Numeric/guidance queries
```

**Impact:** Different query types get optimized retrieval strategies (e.g., comparisons search per-company, numeric queries use stricter reranking).

#### 1.5 Parent Document Retrieval
```python
class ParentExpander:
    # For PDFs: page-level parents
    # For HTML/TXT: section-level parents
    
    # Child chunks used for precise matching
    # Parent content retrieved for full context
```

**Impact:** Small chunks enable precise retrieval; parent expansion provides complete context including tables, footnotes, and units on same page.

#### 1.6 Time Decay for Recency
```python
def apply_time_decay(results, decay_rate=0.1):
    # Exponential decay based on document date
    # Different decay rates per doc_type:
    #   - earnings_call: 0.15 (faster decay)
    #   - 10-K: 0.05 (slower decay)
    #   - press_release: 0.20 (fastest decay)
```

**Impact:** For real-time queries, recent documents are prioritized appropriately.

#### 1.7 Evidence Alignment for Comparisons
```python
def align_evidence_per_company(results, companies, min_per_company=2, max_per_company=4):
    # Ensures balanced evidence coverage across companies
    
def ensure_evidence_coverage(aligned, companies):
    # Reports gaps if any company has insufficient evidence
```

**Impact:** Comparison queries get balanced evidence from all companies, with explicit gap reporting.

#### 1.8 Structured Context Assembly
```python
def _build_structured_context(results):
    # Groups by: Company → Fiscal Year → Section
    # Prioritizes tables
    # Deduplicates using Jaccard similarity (threshold: 0.85)
    
    # Output format:
    # ═══ FLEX ═══
    # ▸ Fiscal Year 2024
    #   [Cash Flow Statement]
    #   📊 (Page 45)
    #   Purchases of property and equipment: $(505) million
```

**Impact:** LLM receives organized context instead of random concatenation, reducing confusion and token waste.

---

## 2. Query Expansion for Financial Terms

### File Modified:
- `backend/rag/retriever.py`
- `backend/rag/assembled_retriever.py`

### Implementation:
```python
CAPEX_SYNONYMS = [
    "capital expenditures",
    "purchases of property and equipment", 
    "additions to PP&E",
    "property, plant and equipment additions",
    "capex",
]

def expand_query(query):
    if "capex" in query.lower():
        return query + " OR " + " OR ".join(CAPEX_SYNONYMS)
```

**Impact:** Catches different terminology used across companies for the same financial metric.

#### 2b. Classic Retriever: LLM Reranking and Parent Expansion (`retriever.py`)

The **classic** retrieval path (when not using assembled mode) also implements RAG-Challenge-2 style behavior in `backend/rag/retriever.py`:

- **LLM Reranking:** `rerank_with_llm()` — retrieves 3× candidates, scores with `gpt-4o-mini`, uses weighted combination `VECTOR_WEIGHT * vector_score + LLM_WEIGHT * llm_score` (default 0.3 / 0.7). Uses `RERANK_SYSTEM_PROMPT` for “support answering” scoring.
- **Parent Document Expansion:** After vector search, collects `parent_id` from child chunks, fetches full parent documents from ChromaDB, and attaches `parent_content` to each child so the generator receives page/section context.
- **Section Boosting:** Auto-boosts sections by query (e.g. CapEx → Cash Flow Statement, Liquidity and Capital Resources; revenue → Income Statement).
- **Year and Recency Boosting:** Boosts by detected fiscal year and “latest/recent” intent.

**Impact:** Non-assembled chat/RAG still benefits from reranking and parent expansion without using the full assembled pipeline.

---

## 3. Structured Document Parsing

### Files Modified:
- `Vector Database/build_chromadb.py`
- `backend/ingestion/processor.py`

### Features:

#### 3.1 PDF Parsing with pdfplumber
```python
def extract_pdf_structured(filepath):
    # Extracts:
    - Text with page boundaries
    - Tables as structured data
    - Headers/sections detection
    - Unit headers (thousands/millions)
```

#### 3.2 HTML Parsing with BeautifulSoup
```python
def extract_html_structured(filepath):
    # Extracts:
    - SEC section headers (Item 1, Item 7, etc.)
    - Tables with row/column structure
    - Paragraph boundaries
```

#### 3.3 TXT/MD File Support
```python
def extract_txt_structured(filepath):
    # Section-based extraction for plain text
    # Markdown header detection
```

**Impact:** Preserves document structure instead of treating as flat text.

---

## 4. Table Serialization

### Files Modified:
- `Vector Database/build_chromadb.py`
- `backend/ingestion/processor.py`

### Implementation:
```python
def serialize_table(table_data, format="markdown"):
    # Formats:
    # 1. Markdown table (for display)
    # 2. Linearized text (for embedding)
    #    "Row 1: Revenue = $5,000M, Growth = 15%"
    
    # Preserves:
    - Column headers
    - Row labels
    - Units from table caption
    - Table title/context
```

**Impact:** Tables become searchable and retrievable, not lost in parsing.

---

## 5. Structure-Aware Chunking

### Files Modified:
- `Vector Database/build_chromadb.py`
- `backend/ingestion/processor.py`

### Strategy:
```python
def structure_aware_chunk(document):
    # Priority order:
    # 1. SEC section headers (Item 1, Item 7, etc.)
    # 2. Title/heading boundaries
    # 3. Paragraph boundaries
    # 4. Table boundaries (keep tables intact)
    # 5. Sentence boundaries (fallback)
    
    # NOT fixed word count chunks
```

**Impact:** Chunks align with document structure, preserving semantic coherence.

---

## 6. Rich Metadata for Chunks

### Files Modified:
- `Vector Database/build_chromadb.py`

### Metadata Fields:
```python
chunk_metadata = {
    "company": "Flex",
    "fiscal_year": "FY24",
    "quarter": "Q2",
    "filing_type": "10-K",
    "section_header": "Item 7. MD&A",
    "section_level": 1,
    "page_num": 45,
    "table_type": "cash_flow",
    "table_name": "Consolidated Statements of Cash Flows",
    "table_context": "(in millions)",
    "parent_id": "page_45",
    "parent_type": "page",
    "is_parent": False,
    "source_file": "flex_10k_2024.htm",
}
```

**Impact:** Enables metadata filtering, boosting, and precise retrieval routing.

---

## 7. Independent Company Collections

### Files Modified:
- `Vector Database/build_chromadb.py`
- `backend/core/database.py`

### Implementation:
```python
# Instead of single collection:
collection = chroma.get_collection("all_documents")

# Per-company collections:
collections = {
    "Flex": chroma.get_collection("flex_documents"),
    "Jabil": chroma.get_collection("jabil_documents"),
    # ...
}

def search_company(company, query):
    return collections[company].query(query)
```

**Impact:** Company-specific queries search smaller, focused indexes—faster and more precise.

---

## 8. Structured Output + Chain of Thought

### Files:
- **Created:** `backend/rag/advanced_prompts.py` (typed answer schemas, comparison pipeline)
- **Modified:** `backend/rag/generator.py` (CoT schemas, prompts, and generation)

---

### 8a. Chain of Thought (CoT) — Full Implementation

Inspired by RAG-Challenge-2’s “structured output prompting with chain-of-thought reasoning.” CoT is implemented in **`backend/rag/generator.py`** so the model reasons step-by-step before the final answer.

#### Reasoning step schema
```python
class ReasoningStep(BaseModel):
    """A single step in the chain of thought reasoning."""
    step: str = Field(description="Description of this reasoning step")
    finding: str = Field(description="What was found/concluded in this step")
```

#### Response schemas with CoT
Every structured answer type includes a **step-by-step analysis** field:

```python
class StructuredAnswer(BaseModel):
    step_by_step_analysis: list[ReasoningStep]  # Chain of thought steps
    reasoning_summary: str
    final_answer: str
    confidence: str
    relevant_sources: list[str]

class NumericAnswer(BaseModel):
    step_by_step_analysis: list[ReasoningStep]  # Steps to extract/validate the number
    raw_value_found: str
    unit_in_source: str
    normalized_value: float
    fiscal_period: str
    final_answer: str
    confidence: str
    source_section: str

class ComparisonAnswer(BaseModel):
    step_by_step_analysis: list[ReasoningStep]  # Steps comparing each company
    company_data: dict
    comparison_result: str
    final_answer: str
    confidence: str
```

#### CoT system prompt
```python
COT_SYSTEM_PROMPT = BASE_SYSTEM_PROMPT + """

=== Chain of Thought Instructions ===

IMPORTANT: You must think step-by-step before providing your final answer.

For each question:
1. First, identify what specific information is being asked
2. Search through the context for relevant data
3. Verify the data matches the question (right company, right period, right metric)
4. Check for potential pitfalls (YTD vs quarterly, unit conversion, negative signs)
5. Formulate your answer with proper units and context

If the exact data is not found:
- State clearly that the specific data was not found
- Do NOT hallucinate or make up numbers
- If similar/related data exists, mention it but clarify it's not exact

Your reasoning steps should be visible in the step_by_step_analysis field."""
```

#### Generation with CoT
- **`generate_structured_response(query, context, ...)`**  
  Uses OpenAI structured output (`response_format=response_schema`). For each response it:
  - Fills `step_by_step_analysis` with `ReasoningStep` objects.
  - Returns a dict with `reasoning` (list of `{step, finding}`), `answer`, `confidence`, `sources`, plus numeric/comparison-specific fields.
- **`generate_with_cot(query, context)`**  
  Wrapper that calls `generate_structured_response` and returns formatted text (and adds a low-confidence note when applicable).
- **`format_structured_response_for_display(result, show_reasoning=True)`**  
  When `show_reasoning` is True, prepends a “Reasoning Steps” section (numbered steps and findings) before the answer, so the UI can show the full chain of thought.

**Impact:** The model is forced to reason step-by-step and expose that reasoning in `step_by_step_analysis`, improving accuracy and debuggability; optional display of reasoning in the UI is supported.

---

### 8b. Additional structured schemas (advanced_prompts.py)

```python
class NumberAnswer(BaseModel):
    reasoning: str  # Chain of thought
    value: Optional[float]
    unit: str
    period: str
    confidence: float
    source_page: Optional[int]

class ComparisonAnswer(BaseModel):
    plan: str  # Pre-answer planning
    companies: List[CompanyData]
    winner: Optional[str]
    summary: str
    caveats: List[str]

# Also: NameAnswer, BooleanAnswer, NamesListAnswer
```

### Answer type detection
```python
def detect_answer_type(query) -> str:
    # Returns: "number" | "name" | "boolean" | "names_list" | "comparison"
    
    # "What was Flex's CapEx?" → "number"
    # "Who is the CEO?" → "name"
    # "Did revenue increase?" → "boolean"
    # "List all subsidiaries" → "names_list"
    # "Compare CapEx across companies" → "comparison"
```

**Impact:** Structured, verifiable outputs with explicit reasoning chains; CoT is implemented in `generator.py` and can be shown in the UI via `format_structured_response_for_display(..., show_reasoning=True)`.

---

## 9. Comparison Query Pipeline

### File Created:
- `backend/rag/advanced_prompts.py`

### Pipeline:
```python
async def answer_comparison_query(query, retriever):
    # Step 1: Rephrase into per-company sub-questions
    sub_questions = rephrase_comparison_query(query)
    # "Compare CapEx" → ["What is Flex's CapEx?", "What is Jabil's CapEx?", ...]
    
    # Step 2: Retrieve & answer each separately
    answers = {}
    for company, question in sub_questions:
        context = retriever.search(question, company_filter=company)
        answers[company] = generate_typed_answer(question, context)
    
    # Step 3: Merge into comparative response
    final = merge_comparative_answers(answers, original_query)
    return final
```

**Impact:** Complex comparisons broken into manageable pieces, then synthesized—more accurate than single-shot answering.

---

## 10. Enhanced Generator Prompts

### File Modified:
- `backend/rag/generator.py`

### Comparison System Prompt:
```python
COMPARISON_SYSTEM_PROMPT = """
For comparison questions, follow this structured plan:

STEP 1 - PLAN:
   - Which companies are being compared?
   - What metric(s) need to be found?
   - What time period(s) are relevant?
   - Which sections should contain this data?

STEP 2 - EXTRACT (for EACH company):
   - Find the relevant metric
   - Note exact value and units
   - Note fiscal period
   - Normalize to millions for comparison

STEP 3 - VALIDATE:
   - Same metric definition?
   - Comparable time periods?
   - Same reporting basis?

STEP 4 - COMPARE:
   - State each company's value
   - Rank from highest to lowest
   - Note limitations or missing data

CRITICAL - HANDLING MISSING DATA:
   - If data NOT found, explicitly state it
   - Do NOT guess or interpolate
   - Address each mentioned company
"""
```

**Impact:** Forces systematic reasoning for complex comparison queries.

---

## 11. Event Extraction System (CapEx Signals)

### File Created:
- `backend/rag/event_extraction.py`

### Event Schema:
```python
class CapExEvent:
    event_id: str
    company: str
    event_date: str
    event_type: EventType  # CAPEX_GUIDANCE, FACILITY_OPEN, M_AND_A, etc.
    amount_value: Optional[float]
    amount_currency: str
    amount_unit: str
    time_horizon: str
    category_bucket_l1: BucketL1  # AI_DC, Traditional, Mixed
    category_bucket_l2: str  # GPU_SERVERS, LIQUID_COOLING, etc.
    geo: str
    confidence: float
    evidence_spans: List[EvidenceSpan]
```

### Two-Stage Extraction:
```python
# Stage 1: Rule-based candidate screening
def detect_candidate_signal(text) -> bool:
    # Keyword patterns for investment signals
    
# Stage 2: LLM structured extraction
def extract_structured_event(text, candidate) -> CapExEvent:
    # Uses Pydantic schema for structured output
```

### Event Deduplication:
```python
def deduplicate_events(events, time_window_days=30):
    # Merges similar events based on:
    # - Same company + event_type + geo
    # - Similar amount (within 10%)
    # - Within time window
    # Prioritizes more authoritative sources
```

**Impact:** Structured extraction of investment signals for tracking and analysis.

---

## 12. CLI Pipeline

### File Created:
- `scripts/pipeline_cli.py`

### Commands:
```bash
# Stage-by-stage pipeline execution
python scripts/pipeline_cli.py download-filings
python scripts/pipeline_cli.py parse-documents
python scripts/pipeline_cli.py build-index [--per-company]
python scripts/pipeline_cli.py run-query "What was Flex's CapEx?" [--assembled]
python scripts/pipeline_cli.py evaluate --questions questions.json
python scripts/pipeline_cli.py full-pipeline
```

### Features:
- Artifact persistence (intermediate outputs saved to disk)
- Skip completed stages
- Verbose logging
- Strategy selection (classic vs assembled retriever)

**Impact:** Reproducible, debuggable pipeline inspired by RAG-Challenge-2's modular approach.

#### 12b. Chat API and Pipeline Integration for Assembled Mode

**Files Modified:** `backend/api/routes/chat.py`, `backend/rag/pipeline.py`

- **Chat request:** `ChatRequest` supports `mode: "rag" | "web" | "hybrid" | "assembled"`, `retrieval_strategy: "auto" | "vector" | "bm25" | "hybrid" | "table"`, and `use_reranking: bool`. When `mode == "assembled"`, the stream and non-stream endpoints call `get_assembled_retriever().search(...)` with parent expansion and optional reranking; response includes `query_analysis` and `strategy_used`.
- **Pipeline:** `process_query()` and `process_query_sync()` accept `mode="assembled"`, `retrieval_strategy`, and `use_reranking`; assembled mode uses `AssembledRetriever`, formats sources from results, and returns `query_analysis` and `retrieval_strategy` in the result dict.

**Impact:** Frontend and other clients can use assembled retrieval and see which strategy was used.

---

## 13. Dependencies Added

### File Modified:
- `backend/requirements.txt`

### New Dependencies:
```
rank-bm25>=0.2.2        # BM25 keyword search
pdfplumber>=0.10.0      # Enhanced PDF parsing
```

---

# Part B: UI/UX Enhancements

## 14. Big 5 AI Investment Tracker

### File Created:
- `frontend/src/app/ai-investments/page.tsx`

### Data Source:
Based on [Futurum Research - AI Capex 2026](https://futurumgroup.com/insights/ai-capex-2026-the-690b-infrastructure-sprint/). Data is served by `GET /api/intelligence/big5-capex` from `backend/api/routes/intelligence.py` (BIG5_AI_CAPEX).

### Data Structures (Frontend):
```tsx
interface Big5Company {
  name: string;
  ticker: string;
  capex_2026_billions: number;
  capex_2025_billions: number;
  yoy_growth_pct: number;
  ai_focus_areas: string[];
  key_metrics: Record<string, number>;
  recent_announcements: string[];
  color: string;
}

interface StargateProject {
  total_investment_billions: number;
  timeline: string;
  partners: string[];
  initial_deployment_billions: number;
  planned_capacity_gw: number;
  locations: string[];
}
```

### Page Structure and Features:
- **Header:** Title "Big 5 AI CapEx Tracker", source attribution, last updated, refresh button.
- **Summary cards:** Total 2026 CapEx (sum of 5 companies), average YoY growth %, Stargate project total ($500B) with timeline and partners.
- **CapEx comparison bar chart (Recharts):** BarChart with 2026 vs 2025 CapEx per company, `Cell` colors from company `color`, Tooltip with growth %.
- **CapEx distribution pie chart:** PieChart of 2026 CapEx share by company.
- **Company selector + detail panel:** Clicking a company shows investment profile: AI focus areas, key metrics, recent announcements, link to external source.
- **ChartDescription** below charts with description, source, lastUpdated.
- **Responsive layout:** Gradient background `from-slate-50 to-slate-100`, Card-based sections.

### API Usage:
- Single fetch on mount: `GET ${API_URL}/api/intelligence/big5-capex`, then `setData(json)` and default `setSelectedCompany(json.companies[0])`.

**Impact:** Provides a dedicated view of hyperscaler AI CapEx (AWS, Google, Microsoft, Meta, Oracle), Stargate project context, and company-level metrics so users can see demand drivers for EMS.

---

## 15. News Monitor

### File Created:
- `frontend/src/app/news-monitor/page.tsx`

### Data Source:
`GET /api/intelligence/news/all` returns `{ press_releases, ocp_news, industry_news }` from `MONITORED_NEWS` in `backend/api/routes/intelligence.py`.

### Data Structures (Frontend):
```tsx
interface PressRelease {
  company: string;
  title: string;
  date: string;
  url: string;
  summary: string;
  category: string;
}

interface OCPNews {
  title: string;
  date: string;
  url: string;
  relevance: string;
  companies_mentioned: string[];
}

interface IndustryNews {
  title: string;
  source: string;
  date: string;
  url: string;
  summary: string;
  relevance: string;
}
```

### Page Structure and Features:
- **Tabs:** `all` | `press` | `ocp` | `industry` — All News, Press Releases, OCP/Standards, Industry News. Icons: Newspaper, Building2, Server, Globe.
- **Company filter:** Dropdown for Flex, Jabil, Celestica, Benchmark, Sanmina. Filters press releases by `company`, OCP news by `companies_mentioned`.
- **Press releases list:** Card per item with company badge (color from `COMPANY_COLORS`), title, date, summary, category, external link.
- **OCP news list:** Title, date, relevance, companies mentioned, link to opencompute.org.
- **Industry news list:** Title, source, date, summary, relevance, link.
- **Loading and error states:** Loading spinner; "Failed to load data" if fetch fails.
- **Refresh button:** Re-fetches `news/all`.

**Impact:** Centralizes official press releases of the 5 EMS companies, OCP-related news (e.g. opencompute.org), and industry-wide AI/investment news in one page with filtering.

---

## 16. Competitor Investments Monitor

### File Created:
- `frontend/src/app/competitor-investments/page.tsx`

### Data Source:
`GET /api/intelligence/competitor-investments` returns `{ competitors, hyperscaler_demand }`. Competitors are derived from `EMS_AI_DYNAMICS["companies"]` with `investment_focus`, `guidance_outlook`, `recent_highlights`, `ai_growth_pct`; hyperscaler_demand is a fixed structure (outlook, drivers, beneficiaries).

### Data Structures (Frontend):
```tsx
interface CompetitorInvestment {
  company: string;
  investment_focus: string[];
  guidance_outlook: string;
  recent_highlights: string[];
  ai_growth_pct: number;
}

interface HyperscalerDemand {
  outlook: string;
  drivers: string[];
  beneficiaries: string[];
}
```

### Page Structure and Features:
- **Hyperscaler Demand banner:** Card with outlook ("Very Strong"), drivers (e.g. Big 5 $675B+, supply-constrained, liquid cooling demand), beneficiaries (Celestica, Jabil, Flex).
- **AI Revenue Growth bar chart (Recharts):** BarChart of `ai_growth_pct` per company, colored by `COMPANY_COLORS`, with ChartDescription.
- **Investment Profile radar chart:** RadarChart (PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar) for investment focus dimensions; one series per company.
- **Company selector + detail panel:** Guidance/outlook (with style from `getOutlookStyle()` — e.g. "Very bullish" → green, "Cautious" → amber), investment focus list, recent highlights list.
- **Investment Comparison Matrix:** Table of companies × investment_focus / guidance / ai_growth_pct for side-by-side comparison.
- **ChartDescription** under charts.

**Impact:** Enables monitoring of competitor investment plans and strategies (CEO/CFO statements, earnings call–style guidance) and hyperscaler demand outlook in one place.

---

## 17. Chart Description Component

### File Created:
- `frontend/src/components/ui/chart-description.tsx`

### Interface and Implementation:
```tsx
interface ChartDescriptionProps {
  description: string;
  source?: string;
  lastUpdated?: string;
}

export function ChartDescription({ description, source, lastUpdated }: ChartDescriptionProps) {
  return (
    <div className="mt-3 pt-3 border-t border-slate-100">
      <div className="flex items-start gap-2">
        <Info className="h-4 w-4 text-slate-400 mt-0.5 flex-shrink-0" />
        <div>
          <p className="text-sm text-slate-600">{description}</p>
          {(source || lastUpdated) && (
            <p className="text-xs text-slate-400 mt-1">
              {source && <span>Source: {source}</span>}
              {source && lastUpdated && <span> • </span>}
              {lastUpdated && <span>Updated: {lastUpdated}</span>}
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
```

### Usage (Applied to Dashboard Charts):
- **AI/Data Center Investment Focus chart:** `description="Percentage of earnings call mentions and SEC filing content focused on AI, data center, and related infrastructure investments across EMS companies."`
- **Sentiment Analysis chart:** `description="Sentiment scores derived from NLP analysis of earnings calls and SEC filings. Higher scores indicate more positive outlook and confidence in company communications."`
- **Documents by Company chart:** `description="Distribution of indexed documents across the 5 tracked EMS companies. Includes SEC filings, earnings calls, press releases, and other public disclosures."`
- **Documents by Type chart:** `description="Breakdown of document types in the knowledge base including 10-K annual reports, 10-Q quarterly filings, earnings call transcripts, and press releases."`

Optional `source` and `lastUpdated` are shown in smaller text below the description when provided.

**Impact:** Meets client requirement that every chart have a clear description so users know what the chart represents and where/when the data comes from.

---

## 18. Dashboard Enhancements

### File Modified:
- `frontend/src/app/dashboard/page.tsx`

### New State and Data Fetching:
```tsx
const [analystQuestions, setAnalystQuestions] = useState<AnalystQuestion[]>([]);
const [emsAIDynamics, setEmsAIDynamics] = useState<EMSAIDynamic[]>([]);

// In fetchAllData (Promise.all):
fetch(`${API_URL}/api/intelligence/default-questions`),
fetch(`${API_URL}/api/intelligence/ems-ai-dynamics`),
```
Response handling: `default-questions` returns `{ questions, categories }` — store `questions` in `analystQuestions`. `ems-ai-dynamics` returns `{ last_updated, companies }` — store `companies` in `emsAIDynamics`.

### New Sections Added to Dashboard:
1. **Analyst Questions (top 6):** Card titled with MessageSquare icon "Latest analyst questions". Renders `analystQuestions.slice(0, 6)`. Each item: question text, category badge, link to `/chat` with query pre-filled (e.g. `?q=${encodeURIComponent(q.question)}` or equivalent). Copy: "Analyst questions from recent EMS earnings calls".
2. **EMS AI Dynamics (5 companies):** Card with Cpu icon. For each company in `emsAIDynamics`: company name, ticker, AI revenue growth %, AI revenue mix %, recent highlights (bullets), investment focus (badges/tags), guidance/outlook. Link to `/ai-investments` for more.
3. **Big 5 AI Investment Summary banner:** Full-width Card with gradient `from-orange-500 to-red-600`. Headline "Big 5 AI CapEx 2026: $675B+", short blurb on hyperscaler demand, CTA "View Big 5 AI CapEx Tracker" linking to `/ai-investments`.

### Chart Descriptions:
- Each of the four main dashboard charts (AI/Data Center focus, Sentiment, Documents by Company, Documents by Type) now has a `<ChartDescription … />` below it with the descriptions listed in §17.

### Interfaces Added:
- `AnalystQuestion`: `id`, `question`, `category`, `complexity`, `companies`.
- `EMSAIDynamic`: `company`, `ticker`, `ai_revenue_growth_pct`, `ai_revenue_mix_pct`, `recent_highlights`, `investment_focus`, `guidance_outlook`.

**Impact:** Dashboard becomes the hub for latest analyst questions (with one-click to chat), EMS AI dynamics at a glance, and a prominent Big 5 CapEx summary with navigation to the dedicated tracker.

---

## 19. Chat Page Updates

### File Modified:
- `frontend/src/app/chat/page.tsx`

### Replaced Suggested Queries with Analyst Questions:
```tsx
const ANALYST_QUESTIONS = [
  { question: "What is the AI/Data Center revenue mix for each company, and how has it changed YoY?", category: "AI Investment" },
  { question: "Compare CapEx guidance across all 5 EMS companies for the current fiscal year", category: "CapEx" },
  { question: "What liquid cooling and power management capabilities are each company developing?", category: "AI Infrastructure" },
  { question: "Which hyperscaler customers are driving AI server demand for EMS companies?", category: "Customers" },
  { question: "What are the gross margin trends for AI/DC vs traditional segments?", category: "Financials" },
  { question: "What manufacturing capacity expansions are planned for AI server production?", category: "Operations" },
];
```

### UI Behavior:
- Suggested questions are rendered as clickable cards/chips. Each shows `question` and a **category badge** (e.g. "AI Investment", "CapEx", "AI Infrastructure").
- Introductory note above the list: e.g. "Analyst questions from recent EMS earnings calls".
- On click: the question text is sent as the user message (e.g. `sendMessage(question)` or `setInput(question)` then submit), so the chat request body includes this as `query`.
- Chat request still uses `POST /api/chat` with `query`, `mode`, `session_id`.

**Impact:** Default prompts in chat align with real analyst questions from earnings calls (comparison, CapEx, AI infrastructure, customers, financials, operations), improving relevance for competitive intelligence use cases.

---

## 20. Navigation Updates

### File Modified:
- `frontend/src/components/layout/Sidebar.tsx`

### navItems Additions:
```tsx
const navItems = [
  // ... existing
  { href: '/ai-investments',    label: 'Big 5 AI CapEx',       icon: TrendingUp,  badge: 'NEW' },
  { href: '/competitor-investments', label: 'EMS Competitors', icon: GitCompare,  badge: 'NEW' },
  { href: '/news-monitor',      label: 'News Monitor',         icon: Newspaper,   badge: 'NEW' },
  // ...
];
```

### Rendering:
- Each item is a `Link` with `href`, icon component, and `label`.
- When `item.badge` is present (e.g. `'NEW'`), a badge is shown: `ml-auto`, styling `bg-purple-500/20 text-purple-400` when not active, `bg-white/20 text-white` when active. This highlights the three new intelligence pages.
- Icons: TrendingUp (Big 5), GitCompare (Competitors), Newspaper (News Monitor).

**Impact:** Users can discover and open the Big 5 AI CapEx tracker, EMS competitor investments, and News Monitor directly from the sidebar with clear "NEW" indicators.

---

# Part C: Infrastructure & Compatibility

## 21. LLM Provider Migration: Anthropic → OpenAI

### Files Modified:
- `backend/core/config.py`
- `backend/analytics/sentiment.py`
- `backend/rag/agentic.py`
- `backend/extraction/ai_extractor.py`

### Config (`backend/core/config.py`):
```python
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = "gpt-4o"        # Main model for response generation
RERANK_MODEL = "gpt-4o-mini" # Reranking (cheaper, faster)
```
- Removed any `ANTHROPIC_API_KEY` / Anthropic-specific env.
- All LLM calls use OpenAI; no Anthropic client remains.

### Sentiment (`backend/analytics/sentiment.py`):
- **Before:** `import anthropic`, `client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)`, `client.messages.create(model=..., messages=...)`, response from `message.content[0].text`.
- **After:** `from openai import OpenAI`, `client = OpenAI(api_key=OPENAI_API_KEY)`, `client.chat.completions.create(model=LLM_MODEL, messages=[{"role": "user", "content": prompt}])`, `response.choices[0].message.content`, `response.usage.total_tokens`.
- Same function signatures and return shape (e.g. JSON with overall_sentiment, sentiment_score, key_themes, etc.) so existing callers (API, dashboard) unchanged.

### Agentic RAG (`backend/rag/agentic.py`):
- **Before:** Anthropic client and `messages.create()` with tool_use handling.
- **After:** `from openai import OpenAI`, `client = OpenAI(api_key=OPENAI_API_KEY)`. `agentic_stream()` builds `openai_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages` and calls `client.chat.completions.create(model=LLM_MODEL, max_tokens=2000, messages=openai_messages)`. Tool-calling loop was simplified: current implementation yields the assistant text and then `done` (no multi-step tool execution in stream) to avoid OpenAI tool-call format differences; `search_documents` remains available for future tool integration.

### AI Extractor (`backend/extraction/ai_extractor.py`):
- **Before:** Anthropic client, `messages.create()` for CapEx extraction.
- **After:** `from openai import OpenAI`, `client = OpenAI(api_key=OPENAI_API_KEY)`. Uses `client.chat.completions.create(model=LLM_MODEL, max_tokens=500, messages=[{"role": "system", "content": CAPEX_EXTRACTION_PROMPT}, {"role": "user", "content": user_prompt}])`. Response parsing remains `response.choices[0].message.content` fed into existing `parse_extraction_response()`.

**Impact:** The platform runs with a single provider (OpenAI). Users only need `OPENAI_API_KEY`; dual-model usage (gpt-4o for generation, gpt-4o-mini for reranking) is documented in config and .env.example.

---

## 22. Benchmark Folder Reference Fix

### File Modified:
- `Vector Database/build_chromadb.py`

### Context:
Ingestion walks company folders and uses a mapping from folder/key names to canonical company names. The user renamed the folder from lowercase `benchmark` to `Benchmark`.

### Changes:
- **COMPANY_FOLDERS:** Key for the company must match the folder name. Updated so the key is `"Benchmark"` (e.g. `"Benchmark": [ ("benchmark_filings", None), ... ]` or equivalent) so the script finds the `Benchmark` directory.
- **COMPANY_DISPLAY:** `"Benchmark": "Benchmark"` so display and metadata use consistent casing.
- **Company detection / pattern logic:** Any branch that checked `company == "benchmark"` or used lowercase for Benchmark was changed to `company in ("Benchmark",)` or `"Benchmark"` so that filenames under the `Benchmark` folder are attributed to Benchmark Electronics.

Exact locations: key in `COMPANY_FOLDERS` (e.g. line ~112), `COMPANY_DISPLAY` (e.g. ~138), and date/company detection for Benchmark HTML (e.g. ~621).

**Impact:** Build script correctly discovers and indexes documents in the renamed `Benchmark` folder and assigns them to Benchmark in ChromaDB metadata.

---

## 23. Intelligence API Endpoints

### File Created:
- `backend/api/routes/intelligence.py`

### File Modified:
- `backend/main.py`: `from backend.api.routes import intelligence as intelligence_router` and `app.include_router(intelligence_router.router, prefix="/api/intelligence", tags=["Competitive Intelligence"])`.

### Data Structures in `intelligence.py`:
- **BIG5_AI_CAPEX:** Dict with `last_updated`, `source`, `total_2026_capex_billions`, `companies` (list of company dicts with name, ticker, capex 2025/2026, yoy_growth_pct, ai_focus_areas, key_metrics, recent_announcements, color), and `stargate_project` (total_investment_billions, timeline, partners, initial_deployment_billions, planned_capacity_gw, locations).
- **DEFAULT_ANALYST_QUESTIONS:** List of dicts with `id`, `question`, `category`, `complexity`, `companies`.
- **EMS_AI_DYNAMICS:** Dict with `last_updated` and `companies` (company, ticker, ai_revenue_growth_pct, ai_revenue_mix_pct, recent_highlights, investment_focus, guidance_outlook).
- **MONITORED_NEWS:** Dict with `press_releases`, `ocp_news`, `industry_news` (each list of items with title, date, url, company/summary/relevance as appropriate).

### Endpoints (all GET):

| Endpoint | Handler | Returns |
|----------|---------|--------|
| `/api/intelligence/big5-capex` | `get_big5_capex()` | Full BIG5_AI_CAPEX |
| `/api/intelligence/big5-capex/summary` | `get_big5_capex_summary()` | total_2026_billions, companies (name, capex_billions, growth_pct), stargate_project |
| `/api/intelligence/default-questions` | `get_default_questions()` | `{ questions, categories }` |
| `/api/intelligence/ems-ai-dynamics` | `get_ems_ai_dynamics()` | EMS_AI_DYNAMICS |
| `/api/intelligence/ems-ai-dynamics/{company}` | `get_company_ai_dynamics(company)` | Single company from EMS_AI_DYNAMICS or 404-style error |
| `/api/intelligence/news/all` | `get_all_news()` | MONITORED_NEWS |
| `/api/intelligence/news/press-releases` | `get_press_releases()` | `{ press_releases }` |
| `/api/intelligence/news/ocp` | `get_ocp_news()` | `{ ocp_news }` |
| `/api/intelligence/news/industry` | `get_industry_news()` | `{ industry_news }` |
| `/api/intelligence/competitor-investments` | `get_competitor_investments()` | `{ competitors, hyperscaler_demand }` (competitors built from EMS_AI_DYNAMICS; hyperscaler_demand with outlook, drivers, beneficiaries) |

Data is currently static (hardcoded) for demo; endpoints are structured so they can later be backed by DB or live feeds.

**Impact:** Single module and URL prefix for all competitive-intelligence UI data (Big 5, analyst questions, EMS AI dynamics, news, competitor investments); frontend uses these exclusively for the new Part B pages and dashboard sections.

---

## 24. Dependencies Fix

### File Modified:
- `backend/requirements.txt`

### Problem:
Strict pins (e.g. `fastapi==0.109.0`) led to `ResolutionImpossible` or `ModuleNotFoundError` when installing on Python 3.13 or with conflicting transitive dependencies.

### Changes:
- Replaced exact version pins with minimum versions, e.g.:
  - `fastapi==0.109.0` → `fastapi>=0.109.0`
  - `uvicorn[standard]==0.27.0` → `uvicorn[standard]>=0.27.0`
  - `pydantic>=2.5.0`, `openai>=1.0.0`, `chromadb>=0.4.0`, `sentence-transformers>=2.3.0`, `rank-bm25>=0.2.2`, `beautifulsoup4>=4.12.0`, `lxml>=5.0.0`, `PyPDF2>=3.0.0`, `pdfplumber>=0.10.0`, `httpx>=0.26.0`, `apscheduler>=3.10.0`, `typing-extensions>=4.9.0`, `openpyxl>=3.1.0`, `python-pptx>=0.6.0`
- No new packages added in this fix; only constraint relaxation.

**Impact:** `pip install -r requirements.txt` succeeds on current Python and avoids resolution failures while keeping a lower bound for compatibility.

---

## 25. Environment Template (`.env.example`)

### File Modified:
- `backend/.env.example`

### Content (relevant section):
```bash
# LLM - OpenAI API (REQUIRED - Main cost ~$20-50/month)
# Get key at: https://platform.openai.com/api-keys
# Uses dual-model strategy:
#   - gpt-4o for response generation (stronger)
#   - gpt-4o-mini for reranking (cheaper, faster)
OPENAI_API_KEY=sk-your-openai-key-here
```

- Removed any `ANTHROPIC_API_KEY` or similar.
- Single required key for LLM: `OPENAI_API_KEY`.
- Comments explain where to get the key and how the two models are used (generation vs reranking).

**Impact:** New clones copy `.env.example` to `.env`, set `OPENAI_API_KEY`, and have the correct variable name and model strategy documented.

---

## 26. Ingestion Processor — Parent Chunk Model

### File Modified:
- `backend/ingestion/processor.py`

### ChunkWithParent Dataclass:
```python
@dataclass
class ChunkWithParent:
    chunk_id: str
    content: str
    parent_id: str
    parent_content: str
    chunk_type: str   # e.g. "child" or "parent"
    metadata: dict = field(default_factory=dict)
```
- Used when producing chunks that participate in parent-document retrieval: child chunks hold a reference to a parent via `parent_id`; at retrieval time, `parent_content` can be filled from the store.

### Structure-Aware Splitting:
- **`_split_by_sections(text)`:** Splits by SEC-style section headers (e.g. Item 1, Item 7) using `SEC_SECTION_PATTERNS`; returns list of `(section_name, content, level)`.
- **`_split_by_paragraphs(text)`:** Splits by `\n\s*\n`, merges short paragraphs up to `MIN_CHUNK_WORDS` so chunks are not too small.
- **`_split_long_paragraph(text, target_words)`:** Splits long paragraphs by sentence when a single paragraph exceeds target size.
- Parent chunks are created per section (or page) with a stable `parent_id`; child chunks (smaller units within that section) store the same `parent_id` in metadata so the retriever can fetch the full parent later.

### Metadata Written to Chunks:
- Each chunk’s metadata includes `parent_id` (string). When the chunk is a child, this points to the parent chunk’s ID in ChromaDB so `retriever.py` and `assembled_retriever.py` can do parent expansion.

**Impact:** Ingestion output is compatible with parent-document expansion in both the classic retriever and the assembled retriever; no change to RAG API contract, only to the shape and metadata of stored chunks.

---

# Summary: Files Changed

## New Files Created (11):
```
backend/api/routes/intelligence.py
backend/rag/assembled_retriever.py
backend/rag/advanced_prompts.py
backend/rag/event_extraction.py
scripts/pipeline_cli.py
frontend/src/app/ai-investments/page.tsx
frontend/src/app/news-monitor/page.tsx
frontend/src/app/competitor-investments/page.tsx
frontend/src/components/ui/chart-description.tsx
CHANGELOG.md
```

## Files Modified (17):
```
backend/main.py
backend/core/config.py
backend/core/database.py
backend/analytics/sentiment.py
backend/rag/agentic.py
backend/rag/generator.py
backend/rag/retriever.py
backend/rag/pipeline.py
backend/api/routes/chat.py
backend/ingestion/processor.py
backend/extraction/ai_extractor.py
backend/requirements.txt
backend/.env.example
frontend/src/app/dashboard/page.tsx
frontend/src/app/chat/page.tsx
frontend/src/components/layout/Sidebar.tsx
Vector Database/build_chromadb.py
```

---

# Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| Backend API | ✅ Running | Port 8001 |
| Frontend | ⚠️ Blocked | Requires Node.js 20+ (current: 19.6.0) |
| ChromaDB | ✅ Connected | Needs document ingestion |

## To Complete Setup:

```bash
# 1. Upgrade Node.js (using nvm, no sudo required)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
source ~/.zshrc
nvm install 20
nvm use 20

# 2. Start frontend
cd frontend && npm run dev

# 3. Ingest documents
cd "Vector Database" && python build_chromadb.py
```

---

*Document generated: February 25, 2026*
*Based on: [RAG-Challenge-2](https://github.com/ilyaRice/RAG-Challenge-2) winning solution*
