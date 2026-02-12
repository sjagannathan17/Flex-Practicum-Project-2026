# FlexPracticum Project Detailed Execution Plan

## Project Overview
This project aims to develop an AI-driven competitive intelligence analysis system for Flex, utilizing NLP technology to analyze the capital expenditure strategies of competitors in the contract manufacturing industry, with a special focus on the balance between NoteAI/data center investments and traditional businesses.

---

## Overall Project Timeline and Phase Division

### Phase 1: Data Aggregation & Cleaning  
**Expected Time**: 2-3 weeks

### Phase 2: NLP Pipeline Development  
**Expected Time**: 3-4 weeks

### Phase 3: Application Construction  
**Expected Time**: 3-4 weeks

### Phase 4: Strategic Analysis & Reporting  
**Expected Time**: 2 weeks

---

## Stage 1 Detailed Steps: Data Aggregation and Cleaning

### 1.1 Data Collection Preparation
**Task List**:
- [ ] Identify the stock codes and official websites of the five companies
- [ ] Identify primary data sources (SEC EDGAR, company Investor Relations Page, financial data platforms)
- [ ] Set up data storage structure and naming conventions
- [ ] Prepare data collection tools and scripts

### 1.2 Earnings Call Transcript Collection
**Goal**: Transcripts for the past 12 quarters (3 years) for each company

**Data Sources**:
- Seeking Alpha (free transcript sections)
- Company official Investor Relations Page
- SEC 8-K filings attachments
- Financial data APIs (e.g., Alpha Vantage, Financial Modeling Prep)

**Collected Content**:
- Complete meeting transcript text
- Date and quarter information
- Management prepared remarks section
- Q&A session transcript
- List of participating analysts and executives

**Data Format**: 
- Original format: PDF/HTML
- Processed format: JSON/CSV + plain text

### 1.3 Investor Day Presentation Collection
**Data Sources**:
- Company Investor Relations Page
- SEC 8-K filings
- Investor conference websites

**Collected Content**:
- Presentation PDF
- Video transcript (if available)
- Accompanying press release

### 1.4 SEC Filings Collection
**Target Filing Types**:
- **10-K** (annual reports): past 3 years
- **10-Q** (quarterly reports): past 12 quarters
- **8-K** (significant event reports): related past 3 years

**Key Sections to Extract**:
- Item 1: Business (business description)
- Item 1A: Risk Factors
- Item 7: MD&A (Management Discussion and Analysis)
- Item 8: Financial Statements (especially CapEx data)
- Significant announcements and press releases

**Data Source**: SEC EDGAR system (https://www.sec.gov/edgar)

### 1.5 Press Release and Announcement Collection
**Key Topics**:
- New factory openings or expansions
- Major equipment purchases
- M&A activity
- Strategic partnerships
- Technology investments and R&D centers
- Liquid cooling technology, AI infrastructure-related investments

**Data Sources**:
- Company news centers
- PR Newswire
- Business Wire
- Google News Search

### 1.6 Analyst Report Collection (Optional)
**Data Sources**:
- University library databases
- Free analyst report websites
- Investment platforms (e.g., Seeking Alpha)

### 1.7 Data Cleaning and Standardization
**Cleaning Tasks**:
1. **Text extraction**: Extract plain text from PDF/HTML
2. **Format standardization**: Normalize date formats, company names, currency units
3. **Deduplication**: Remove duplicate content
4. **Structuring**: Convert unstructured text into structured JSON format
5. **Metadata addition**: Add metadata such as company name, document type, date, source, etc.
6. **Quality check**: Verify data integrity and accuracy

**Output Format Example**:
```json
{
  "company": "Jabil Inc.",
  "document_type": "earnings_call",
  "date": "2024-Q3",
  "quarter": "Q3",
  "fiscal_year": 2024,
  "source_url": "https://...",
  "sections": {
    "prepared_remarks": "...",
    "qa_session": "...",
    "participants": ["CEO Name", "CFO Name"]
  },
  "raw_text": "...",
  "metadata": {
    "word_count": 5000,
    "collection_date": "2025-01-25"
  }
}
```

---

## Stage 2 Detailed Steps: NLP Pipeline Development

### 2.1 Environment Setup
**Tech Stack**:
- Python 3.9+
- LangChain / LlamaIndex (RAG framework)
- OpenAI API / Anthropic Claude / Llama
- spaCy / NLTK (text preprocessing)
- Pandas (data processing)

### 2.2 Entity Extraction Development
**Target Entity Types**:
- Company names and subsidiaries
- Geographic locations (factories, data center locations)
- Technology types (liquid cooling, AI chips, servers, etc.)
- Financial figures (investment amounts, CapEx numbers)
- Time expressions (project timelines)
- Partner names

**Methods**:
- Use pretrained NER models
- Develop custom prompts for LLM extraction
- Establish entity recognition rule base

### 2.3 CapEx Event Extraction
**Event Type Definition**:
1. **Factory Opening/Expansion**
2. **Equipment Purchase**
3. **M&A Activity**
4. **R&D Investment**
5. **Technology Upgrade**
6. **Partnership**

**Information to Extract**:
- Event type
- Investment amount
- Geographic location
- Timeline
- Business domain (AI/data center vs traditional)
- Related technologies

### 2.4 Investment Classification System
**Classification Dimension 1: Business Type**
- **AI/Data Center**: 
  - Hyperscale data centers
  - AI training/inference infrastructure
  - Liquid cooling technology
  - High-performance computing (HPC)
  - GPU/AI accelerator manufacturing
  
- **Traditional/Non-AI**:
  - Medical equipment manufacturing
  - Automotive electronics
  - Industrial automation
  - Aerospace
  - Consumer electronics

**Classification Dimension 2: Investment Type**
- Capital-intensive (CapEx)
- Operational investment (OpEx)
- Strategic investment

### 2.5 Sentiment Analysis
**Analysis Goals**:
- Management’s optimism about AI/data center business
- Confidence level in traditional business
- Expression of risks and concerns
- Perceived competitive pressures

**Methods**:
- Use LLMs for contextual sentiment analysis
- Develop domain-specific sentiment lexicons
- Quantify sentiment scores (from -1 to +1)

### 2.6 Prompt Engineering
**Develop Dedicated Prompt Templates**:
1. CapEx event extraction prompts
2. Investment classification prompts
3. Sentiment analysis prompts
4. Strategic intent recognition prompts
5. Risk factors extraction prompts

### 2.7 Pipeline Testing and Validation
- Test pipeline on sample data
- Manually verify extraction accuracy
- Iteratively improve prompts and rules
- Establish quality metrics (precision, recall)

---

## Phase 3 Detailed Steps: Application Building

### 3.1 Technical Architecture Design
**System Components**:
1. **Data Storage Layer**: Vector database + relational database
2. **NLP Processing Layer**: LLM API integration
3. **RAG Engine**: Retrieval and generation system
4. **Web Application Layer**: User interface
5. **API Layer**: Backend services

**Technology Choices**:
- **Vector Database**: Pinecone / Weaviate / ChromaDB / FAISS
- **Relational Database**: PostgreSQL / SQLite
- **Backend Framework**: FastAPI / Flask
- **Frontend Framework**: React / Streamlit / Gradio
- **Deployment**: Docker + Cloud services (AWS/Azure/GCP) or on-premises

### 3.2 Vector Database Implementation
**Steps**:
1. Choose embedding model (OpenAI embeddings / Sentence Transformers)
2. Document chunking strategy (chunk size, overlap)
3. Create document index
4. Implement semantic search functionality
5. Optimize retrieval performance

**Index Structure**:
- Partitioned by company
- Partitioned by document type
- Partitioned by time range
- Metadata filtering

### 3.3 RAG System Development
**Retrieval Strategies**:
- Semantic similarity search
- Hybrid search (semantic + keyword)
- Re-ranking
- Context window optimization

**Generation Strategies**:
- Source citation
- Multi-document synthesis
- Confidence scoring
- Answer verification

### 3.4 Chatbot Interface Development
**Feature Requirements**:
1. Natural language query input
2. Real-time response generation
3. Source citation display
4. Conversation history management
5. Query suggestions and autocomplete
6. Result export functionality

**Example Queries**:
- "Which competitors are building liquid cooling capabilities in North America?"
- "How much has Jabil invested in AI data centers over the past two years?"
- "Compare CapEx allocation strategies between Celestica and Sanmina"
- "Identify all companies' investment trends in the medical device sector"

### 3.5 Visualization Dashboard
**Visualization Components**:
1. **Investment Distribution Pie Chart**: AI vs traditional business
2. **Time Series Chart**: CapEx trends
3. **Geographical Heatmap**: Investment geographic distribution
4. **Competitor Comparison**: Side-by-side comparison
5. **Risk Matrix**: AI dependency assessment

### 3.6 User Authentication and Permissions
- Basic login system
- Query log transcript
- Usage analytics

### 3.7 Testing and Deployment
- Unit testing
- Integration testing
- User acceptance testing
- Performance testing
- Deployment to production environment

---

## Stage 4 Detailed Steps: Strategic Analysis and Reporting

### 4.1 Key Strategic Issues Analysis
**Core Questions List**:
1. What is each competitor’s investment ratio in AI/data centers vs. traditional business?
2. Who is the most aggressive investor in liquid cooling technology?
3. Which geographic regions are investment hotspots?
4. How are competitors’ partner networks structured?
5. What is the industry’s overall dependence on AI growth?
6. Is there a risk of overinvestment/bubble?
7. Where are Flex’s "white space" opportunities?
8. What are competitors’ risk exposures?

### 4.2 In-depth Queries Using RAG Tool
- Systematic queries on each strategic question
- Collect and verify answers
- Cross-reference multiple sources
- Quantify findings

### 4.3 Data Integration and Insight Extraction
**Analysis Dimensions**:
1. **Investment Intensity**: Absolute amounts and relative ratios
2. **Strategic Direction**: Growth areas vs. contraction areas
3. **Time Trends**: Investment acceleration or deceleration
4. **Geographic Strategy**: Regional expansion patterns
5. **Technology Focus**: Specific technology investments
6. **Collaboration Models**: Strategic alliances and partnerships

### 4.4 Risk Assessment Matrix Development
**Assessment Dimensions**:
- **X-axis**: AI/data center investment proportion
- **Y-axis**: Investment growth rate
- **Bubble Size**: Total CapEx scale
- **Color**: Risk level

**Risk Classification**:
- 🟢 Low Risk: Balanced investment portfolio
- 🟡 Medium Risk: Moderate AI tilt
- 🔴 High Risk: Over-reliance on AI growth

### 4.5 Competitive Landscape Report Writing
**Report Structure**:
1. **Executive Summary**
   - Key findings
   - Primary recommendations
   
2. **Industry Overview**
   - Market trends
   - Technology drivers
   
3. **Competitor Deep Dive** (each company)
   - Jabil Inc.
   - Celestica
   - Benchmark Electronics
   - Sanmina Corporation
   - Flex (benchmark)
   
4. **Comparative Analysis**
   - Investment strategy comparison
   - Strengths and weaknesses
   
5. **Risk Assessment**
   - Market overheating risk
   - Individual company risk exposure
   
6. **Strategic Recommendations**
   - Flex’s opportunity areas
   - Investment balance suggestions
   - Risk mitigation strategies

### 4.6 Presentation Preparation
**Format**: PowerPoint/Google Slides

**Content**:
- Visualized data
- Key charts and graphics
- Clear narrative flow
- Actionable recommendations

### 4.7 Final Deliverables
**Delivery Checklist**:
- ✅ Functional web application (local or cloud-based)
- ✅ Complete source code repository (GitHub)
- ✅ Technical documentation and user manual
- ✅ Strategic landscape report (PDF)
- ✅ Presentation (PPT)
- ✅ Risk assessment matrix (visualized)
- ✅ Data update guide

---

## Project filings Folder Structure

```
flex-practicum/
├── README.md                          # Project overview and quick start
├── docs/                              # Documentation directory
│   ├── project_requirements.md        # Project requirements document
│   ├── data_collection_guide.md       # Data collection guide
│   ├── api_documentation.md           # API documentation
│   └── user_manual.md                 # User manual
│
├── data/                              # Data directory
│   ├── raw/                           # Raw data
│   │   ├── jabil/                     # Jabil company data
│   │   │   ├── earnings_calls/       # Earnings calls
│   │   │   ├── sec_filings/          # SEC filings
│   │   │   ├── investor_presentations/ # Investor presentations
│   │   │   └── news/                  # News and announcements
│   │   ├── celestica/
│   │   ├── benchmark/
│   │   ├── sanmina/
│   │   └── flex/
│   │
│   ├── processed/                     # Processed data
│   │   ├── cleaned_text/              # Cleaned text
│   │   ├── structured_json/           # Structured JSON
│   │   └── embeddings/                # Vector embeddings
│   │
│   └── analysis/                      # Analysis results
│       ├── capex_events.csv           # CapEx event extraction
│       ├── investment_classification.csv # Investment classification
│       └── sentiment_scores.csv       # Sentiment analysis results
│
├── src/                               # Source code
│   ├── data_collection/               # Data collection scripts
│   │   ├── sec_scraper.py            # SEC filings scraper
│   │   ├── earnings_call_scraper.py  # Earnings call scraper
│   │   └── news_scraper.py           # News scraper
│   │
│   ├── data_processing/               # Data processing
│   │   ├── text_cleaner.py           # Text cleaning
│   │   ├── pdf_extractor.py          # PDF extraction
│   │   └── data_normalizer.py        # Data normalization
│   │
│   ├── nlp/                           # NLP pipeline
│   │   ├── entity_extraction.py      # Entity extraction
│   │   ├── capex_event_detector.py   # CapEx event detection
│   │   ├── investment_classifier.py  # Investment classification
│   │   └── sentiment_analyzer.py     # Sentiment analysis
│   │
│   ├── rag/                           # RAG system
│   │   ├── vector_store.py           # Vector database
│   │   ├── retriever.py              # Retriever
│   │   ├── generator.py              # Generator
│   │   └── rag_pipeline.py           # RAG pipeline
│   │
│   ├── app/                           # Web application
│   │   ├── backend/                  # Backend
│   │   │   ├── main.py               # FastAPI main filings
│   │   │   ├── api/                  # API routes
│   │   │   └── models/               # Data models
│   │   │
│   │   └── frontend/                 # Frontend
│   │       ├── src/                  # React source code
│   │       ├── public/               # Static assets
│   │       └── package.json
│   │
│   └── analysis/                      # Analysis scripts
│       ├── strategic_analysis.py     # Strategic analysis
│       ├── risk_assessment.py        # Risk assessment
│       └── visualization.py          # Visualization
│
├── notebooks/                         # Jupyter notebooks
│   ├── data_exploration.ipynb        # Data exploration
│   ├── nlp_experiments.ipynb         # NLP experiments
│   └── analysis_visualization.ipynb  # Analysis visualization
│
├── reports/                           # Report outputs
│   ├── strategic_landscape_report.pdf # Strategic landscape report
│   ├── presentation.pptx             # Presentation
│   └── risk_assessment_matrix.png    # Risk matrix
│
├── tests/                             # Tests
│   ├── test_data_processing.py
│   ├── test_nlp.py
│   └── test_rag.py
│
├── config/                            # Configuration filings
│   ├── config.yaml                   # Main configuration
│   └── prompts.yaml                  # Prompt templates
│
├── requirements.txt                   # Python dependencies
├── .env.example                       # Environment variables example
├── .gitignore
└── docker-compose.yml                 # Docker configuration
```

## Today's Task: Data Collection for Five Companies

### Target Company List
1. **Jabil Inc.** (Stock Symbol: JBL)  
2. **Celestica Inc.** (Stock Symbol: CLS)  
3. **Benchmark Electronics** (Stock Symbol: BHE)  
4. **Sanmina Corporation** (Stock Symbol: SANM)  
5. **Flex Ltd.** (Stock Symbol: FLEX)  

### Focus for Today's Collection  
Collect the following information for each company:

#### Basic Information
- Full Company Name  
- Stock Symbol  
- Official Website  
- Investor Relations Page URL  
- Headquarters Location  
- Business Overview  

#### Earnings Call Transcripts  
- Transcripts for the past 12 quarters (2022 Q1 - 2024 Q4)  
- Source Links  
- Download or Save Transcripts  

#### SEC Filings  
- Most recent 3 years of 10-K reports (2021, 2022, 2023)  
- Most recent 12 quarterly 10-Q reports  
- Related 8-K material event reports  

#### Investor Presentations  
- Recent Investor Day presentations  
- Quarterly earnings presentations  

#### News and Announcements  
- Major press releases from the past 3 years  
- Announcements regarding CapEx, factory expansions, and technology investments  

---

## Best Practices for Data Collection

### 1. Data Source Prioritization
1. **Official Sources** (Highest Priority)
   - Company Investor Relations Page
   - SEC EDGAR
   
2. **Reliable Third-Party Platforms**
   - Seeking Alpha
   - Yahoo Finance
   - Financial Modeling Prep
   
3. **News Aggregators**
   - Google News
   - Business Wire
   - PR Newswire

### 2. Data Quality Control
- ✅ Verify the reliability of data sources
- ✅ Check dates and versions
- ✅ Ensure text completeness
- ✅ Transcript collection time and method
- ✅ Retain backups of original filings

### 3. Legal and Ethical Considerations
- ✅ Use only publicly available information
- ✅ Comply with website terms of use
- ✅ Respect robots.txt
- ✅ Reasonable crawling rate
- ✅ Properly cite sources

### 4. Organization and Naming Conventions
**filings naming format**:
- Earnings calls: `{company}_{YYYY}_Q{Q}_earnings_call.txt`
- SEC filings: `{company}_{YYYY}_{filing_type}.pdf`
- Presentations: `{company}_{YYYY_MM_DD}_investor_presentation.pdf`
- News: `{company}_{YYYY_MM_DD}_{topic}.txt`

**Examples**:
- `jabil_2024_Q3_earnings_call.txt`
- `celestica_2023_10K.pdf`
- `benchmark_2024_06_15_investor_day.pdf`

---

## Technical Tools and Resources

### Data Collection Tools
- **Python Libraries**: requests, BeautifulSoup, Selenium, scrapy
- **SEC API**: sec-api.io, sec-edgar-downloader
- **Financial Data APIs**: yfinance, Alpha Vantage, Financial Modeling Prep
- **PDF Processing**: PyPDF2, pdfplumber, tabula-py

### Development Tools
- **IDE**: VS Code, PyCharm
- **Version Control**: Git, GitHub
- **Environment Management**: conda, venv
- **Notebooks**: Jupyter Lab

### NLP and ML Tools
- **LLM APIs**: OpenAI, Anthropic, Llama
- **RAG Frameworks**: LangChain, LlamaIndex
- **Vector Databases**: Pinecone, Weaviate, ChromaDB
- **NLP Libraries**: spaCy, NLTK, Hugging Face Transformers

---

## Project Success Metrics

### Data Quality Metrics
- Data completeness: >95% of target documents collected
- Data accuracy: <5% extraction error rate
- Data timeliness: covering past 3 years

### NLP Performance Metrics
- Entity extraction accuracy: >90%
- CapEx event recognition recall: >85%
- Classification accuracy: >90%

### Application Performance Metrics
- Query response time: <5 seconds
- Answer accuracy: >85% (manual evaluation)
- Source citation rate: 100%

### Business Value Metrics
- Quality of strategic insights (qualitative evaluation)
- Effectiveness of decision support
- User satisfaction

---

## Risks and Mitigation Strategies

### Data Collection Risks
- **Risk**: Data unavailable or visits restricted
- **Mitigation**: Use multiple data sources, prepare alternatives

### Technical Risks
- **Risk**: API limitations or costs
- **Mitigation**: Use open-source alternatives, optimize API calls

### Time Risks
- **Risk**: Project delays
- **Mitigation**: Deliver in phases, prioritize core features

### Quality Risks
- **Risk**: Insufficient NLP accuracy
- **Mitigation**: Iterative improvements, manual verification of key findings

---

## Next Steps

### Immediate Actions (Today)
1. ✅ Create GitHub repository
2. ✅ Establish project filings folder structure
3. ✅ Start collecting basic information of five companies
4. ✅ Collect SEC filings links
5. ✅ Download Recent earnings call transcripts

### This Week's Actions
1. Complete all raw data collection
2. Start data cleaning and standardization
3. Set up development environment
4. Begin NLP pipeline prototype development

### This Month's Actions
1. Complete Phase 1 and Phase 2
2. Start RAG system development
3. Prototype application demonstration

---

## Contact and Support
- Project Documentation: GitHub Wiki
- Issue Tracking: GitHub Issues
- Code Review: Pull Requests