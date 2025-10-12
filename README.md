# Workforce Intelligence NLP

Natural language processing and sentiment analysis applied to employment, labor market, and corporate workforce data.

## Overview

This repository demonstrates NLP techniques for extracting insights from employment-related text data. The projects analyze government labor statistics, Federal Reserve policy statements, corporate filings, and Treasury announcements to understand workforce trends and economic sentiment.

**Relevance to Workforce Intelligence:** These methods are directly applicable to HR analytics, employment record standardization, and tracking workforce composition trends across organizations.

## Projects

### 1. BLS Employment Sentiment Analysis
**File:** `_nlp_bls.py`

Analyzes Bureau of Labor Statistics data to create composite employment sentiment scores.

**Data Sources:**
- Unemployment rate changes
- Nonfarm payroll trends  
- Average hourly earnings growth
- Labor force participation rates
- U-6 underemployment metrics
- JOLTS job openings data

**Key Features:**
- Multi-indicator sentiment aggregation with dynamic weighting
- Time series visualization of employment sentiment
- Configurable sensitivity thresholds for classification
- Direct API integration with BLS public data

**Applications:** Understanding labor market health, tracking employment trends, workforce sentiment analysis

---

### 2. FOMC Monetary Policy Analysis  
**File:** `_nlp_fomc.py`

Applies transformer-based NLP to Federal Reserve FOMC statements to classify monetary policy sentiment.

**Techniques:**
- **FinBERT** pre-trained transformer model for financial text
- Web scraping of official Fed statements
- Time series tracking of policy stance changes
- Automated sentiment classification (positive/negative/neutral)

**Key Features:**
- Scrapes FOMC statements from Federal Reserve website
- Applies domain-specific financial language model
- Tracks policy sentiment evolution over time
- Production-ready error handling and validation

**Applications:** Economic indicator analysis, policy stance detection, macro trend identification

---

### 3. SEC 8-K Corporate Filing Analysis
**File:** `_nlp_sec_8k.py`

Extracts and analyzes sentiment from corporate SEC 8-K filings (material event disclosures).

**Techniques:**
- Custom document embeddings (TF-IDF vectorization)
- Domain-specific sentiment lexicons for corporate events
- Entity resolution (ticker symbol → CIK mapping)
- Large-scale document processing pipeline

**Key Features:**
- Automated scraping of SEC EDGAR database
- Ticker-to-CIK entity resolution
- Custom corporate sentiment vocabulary (100+ terms)
- Date-range filtering for temporal analysis
- Sentiment classification: positive/negative/neutral events

**Applications:** Corporate event analysis, M&A sentiment, earnings impact assessment

**Corporate Sentiment Lexicon Examples:**
- Positive: growth, acquisition, synergies, breakthrough, market share, dividend
- Negative: restructuring, layoff, litigation, impairment, writedown, suspension

---

### 4. Treasury Fiscal Policy Analysis
**File:** `_nlp_treasury.py`

Classifies U.S. Treasury press releases as expansionary vs. contractionary fiscal policy.

**Techniques:**
- Custom embedding-based sentiment axes
- Domain-specific fiscal policy vocabulary
- Semantic similarity scoring
- Multi-era document scraping (spanning multiple Treasury Secretaries)

**Key Features:**
- Scrapes Treasury Department press releases (2014-2024)
- Custom fiscal policy lexicon (expansionary vs. contractionary terms)
- Embedding-based semantic analysis
- Handles multiple URL patterns and website structure changes

**Applications:** Fiscal policy tracking, government spending analysis, economic policy stance detection

**Fiscal Policy Lexicon Examples:**
- Expansionary: stimulus, investment, infrastructure, relief, jobs, funding
- Contractionary: deficit reduction, austerity, spending cuts, fiscal discipline

---

## Technical Approach

### NLP Techniques Demonstrated
- **Text preprocessing:** Cleaning, tokenization, regex pattern matching
- **Document embeddings:** TF-IDF vectorization, semantic vector spaces
- **Transformer models:** FinBERT for financial domain text
- **Sentiment analysis:** Rule-based lexicons + ML-based classification
- **Named entity extraction:** Date parsing, ticker symbols, government IDs
- **Custom vocabulary development:** Domain-specific sentiment lexicons
- **Web scraping:** BeautifulSoup, requests, rate limiting, error handling

### Machine Learning & Statistics
- Pre-trained transformers (BERT architecture)
- TF-IDF feature extraction
- Cosine similarity for semantic analysis
- Time series visualization
- Multi-indicator aggregation with dynamic weighting
- Threshold-based classification with adjustable sensitivity

### Data Engineering
- API integration (BLS, SEC EDGAR)
- Web scraping at scale with rate limiting
- Multi-source data aggregation
- Date-range filtering and temporal analysis
- Error handling and validation
- Modular, reusable pipeline architecture

---

## Relevance to Workforce Intelligence

These projects demonstrate core capabilities needed for HR analytics and employment data processing:

| Capability | Demonstrated By | Application to HR Data |
|------------|-----------------|------------------------|
| **Entity standardization** | Ticker→CIK mapping, date normalization | Job title standardization, company name resolution |
| **Document classification** | 8-K sentiment, policy stance detection | Resume classification, job posting categorization |
| **Custom vocabularies** | Corporate/fiscal lexicons (200+ terms) | HR-specific terminology, skill taxonomies |
| **Time series analysis** | Employment sentiment trends | Workforce composition changes, hiring trends |
| **Large-scale text processing** | Multi-year document scraping | Processing millions of employment records |
| **Multi-source integration** | BLS API + web scraping | Aggregating employment data across platforms |

**Key Insight:** Employment records are structured text data requiring many of the same NLP techniques demonstrated here—entity extraction, sentiment classification, temporal trend analysis, and cross-source standardization.

---

## Technologies

**Core:**
- Python 3.8+
- pandas, numpy
- scikit-learn
- matplotlib

**NLP:**
- transformers (HuggingFace)
- BeautifulSoup4
- requests
- regex

**Domain-Specific:**
- FinBERT (financial language model)
- BLS API (Bureau of Labor Statistics)
- SEC EDGAR API

---

## Sample Output

### Employment Sentiment Over Time
Employment sentiment scores derived from 6 BLS indicators (unemployment, payrolls, wages, participation, underemployment, job openings):

- **Positive sentiment periods:** Strong job growth, rising wages, declining unemployment
- **Negative sentiment periods:** Payroll declines, participation drops, rising underemployment  
- **Neutral periods:** Mixed signals across indicators

### FOMC Policy Sentiment
Transformer-based classification of Federal Reserve statements:
- Tracks evolution from accommodative → neutral → restrictive policy stances
- Identifies turning points in monetary policy

### Corporate Event Sentiment  
Analysis of material corporate events via 8-K filings:
- Positive: acquisitions, earnings beats, product launches
- Negative: restructurings, regulatory issues, executive departures

---
