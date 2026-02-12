# Flex Practicum: Strategic CapEx Intelligence in Contract Manufacturing

## Project Overview

This project aims to develop a competitive intelligence analysis system for Flex Inc. based on Natural Language Processing (NLP) and Retrieval-Augmented Generation (RAG) technologies. The system will analyze the capital expenditure (CapEx) strategies of major competitors in the contract manufacturing industry, with particular focus on the balance between AI/data center investments and traditional business, helping Flex make more informed strategic investment decisions.

## Background

Flex (formerly Flextronics) is a global leader in diversified manufacturing services and supply chain solutions. With the surge in infrastructure construction driven by generative AI, Flex is expanding its investment portfolio in power, computing, and liquid cooling technologies. This project helps Flex identify market opportunities, assess risk exposure, and optimize its investment strategy by analyzing competitors' investment patterns.

## Target Companies for Analysis

1. **Jabil Inc.** (JBL) - Focus on operational expansion and data center strategy
2. **Celestica Inc.** (CLS) - Analyze the shift toward hardware platform solutions
3. **Benchmark Electronics** (BHE) - Investigate investment balance between high-performance computing and traditional business
4. **Sanmina Corporation** (SANM) - Review vertical integration strategy and server ecosystem investments
5. **Flex Ltd.** (FLEX) - Serve as control group for benchmarking

## Project Objectives

1. **Quantify Competitor Focus**: Determine the proportion of investment flowing to "AI/Data Center" vs. "Traditional" sectors
2. **Identify Risk Exposure**: Assess collective industry overexposure to AI spending trends
3. **Enable Natural Language Discovery**: Create an intuitive RAG application supporting complex strategic queries

## Project Phases

### Phase 1: Data Aggregation & Cleaning (Current Phase)
- Collect earnings call transcripts from the past 12 quarters
- Obtain SEC filings (10-K, 10-Q, 8-K)
- Gather investor presentations and news releases
- Clean and standardize text data

### Phase 2: NLP Pipeline Development
- Entity extraction and sentiment analysis
- CapEx event detection
- Investment classification (AI/Data Center vs. Traditional Business)
- Prompt engineering and model optimization

### Phase 3: Application Development
- Implement vector database
- Develop RAG retrieval and generation system
- Build chatbot interface
- Create visualization dashboard

### Phase 4: Strategic Analysis & Reporting
- Conduct strategic queries using RAG tools
- Generate competitive landscape report
- Create risk assessment matrix
- Provide investment recommendations

## Project Deliverables

1. **Competitive Intelligence Application**: Functional web application supporting natural language queries
2. **Source Code and Documentation**: Complete Python code repository with usage documentation
3. **Strategic Landscape Report**: Presentation summarizing competitor investment posture
4. **Risk Assessment Matrix**: Visualization showing AI dependency of each competitor

## Technology Stack

### Data Collection & Processing
- Python 3.9+
- BeautifulSoup, Selenium (web scraping)
- PyPDF2, pdfplumber (PDF processing)
- Pandas (data processing)

### NLP & Machine Learning
- OpenAI API / Anthropic Claude
- LangChain / LlamaIndex (RAG framework)
- spaCy, NLTK (text processing)
- Sentence Transformers (embedding models)

### Database & Storage
- Pinecone / ChromaDB (vector database)
- PostgreSQL / SQLite (relational database)

### Web Application
- FastAPI (backend)
- React / Streamlit (frontend)
- Docker (containerization)

## Project Structure

```
flex-practicum/
├── README.md                          # Project overview
├── docs/                              # Documentation
├── data/                              # Data directory
│   ├── raw/                           # Raw data (organized by company)
│   ├── processed/                     # Processed data
│   └── analysis/                      # Analysis results
├── src/                               # Source code
│   ├── data_collection/               # Data collection scripts
│   ├── data_processing/               # Data processing
│   ├── nlp/                           # NLP pipeline
│   ├── rag/                           # RAG system
│   ├── app/                           # Web application
│   └── analysis/                      # Analysis scripts
├── notebooks/                         # Jupyter notebooks
├── reports/                           # Report outputs
├── tests/                             # Tests
└── config/                            # Configuration files
```

## Quick Start

### Environment Setup

```bash
# Clone repository
git clone https://github.com/xcai2/flex-practicum.git
cd flex-practicum

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env file to add API keys
```

### Data Collection

```bash
# Run data collection scripts
python src/data_collection/sec_scraper.py
python src/data_collection/earnings_call_scraper.py
```

### Run Application

```bash
# Start backend
cd src/app/backend
uvicorn main:app --reload

# Start frontend (in another terminal)
cd src/app/frontend
npm install
npm start
```

## Data Sources

- **SEC EDGAR**: Official financial filings
- **Company Investor Relations Pages**: Earnings call transcripts and presentations
- **Seeking Alpha**: Supplementary earnings call transcripts
- **Business Wire / PR Newswire**: News releases

## Development Progress

- [x] Project planning and requirements analysis
- [x] GitHub repository creation and structure setup
- [x] Phase 1: Data collection (in progress)
- [ ] Phase 2: NLP pipeline development
- [ ] Phase 3: Application development
- [ ] Phase 4: Strategic analysis and reporting

## Contributing

This project is an academic practicum project. To contribute:

1. Fork this repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is for educational and research purposes only.

## Contact

Project Maintainer: [Your Name]
Project Link: https://github.com/xcai2/flex-practicum

## Acknowledgments

- Flex Inc. for providing project background and requirements
- Faculty advisors and peers for their support
- Open source community for tools and resources
