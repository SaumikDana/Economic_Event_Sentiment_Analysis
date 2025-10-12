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

**Sample Output:**

![BLS Employment Sentiment Classification](examples/bls_sentiment_classification.png)
*Employment sentiment classification (2014-2024) showing COVID-19 impact in early 2020*

![BLS Employment Sentiment Score](examples/bls_sentiment_score.png)
*Continuous sentiment scores revealing sharp decline during pandemic and subsequent recovery*

**Key Insights from Visualization:**
- **COVID-19 Detection:** Dramatic negative sentiment spike in March-April 2020, with sentiment score plunging from 0 to -9 (unprecedented in the dataset)
- **Recovery Pattern:** Rapid V-shaped recovery through late 2020-2021 as labor markets rebounded
- **Return to Normalcy:** Sentiment stabilized around neutral by 2021-2022, indicating labor market healing
- **Long-term Baseline:** Pre-pandemic period (2014-2019) shows consistently positive sentiment with minimal volatility
- **Methodology Validation:** The model successfully captures the most significant labor market disruption in modern history, demonstrating robust signal detection

**Applications:** Understanding labor market health, tracking employment trends, workforce sentiment analysis

---

### 2. FOMC Monetary Policy Analysis  
**File:** `_nlp_fomc.py`

Applies transformer-based NLP to Federal Reserve FOMC statements to classify monetary policy sentiment.

**Techniques:**
- **FinBERT-FOMC** pre-trained transformer model for Federal Reserve text
- Web scraping of official Fed statements
- Time series tracking of policy stance changes
- Automated sentiment classification (positive/negative/neutral)

**Key Features:**
- Scrapes FOMC statements from Federal Reserve website
- Applies domain-specific financial language model
- Tracks policy sentiment evolution over time
- Production-ready error handling and validation

**Sample Output:**

![FOMC Sentiment Classification](examples/fomc_sentiment_classification.png)
*FOMC statement sentiment classification (2014-2024) using FinBERT-FOMC transformer model*

![FOMC Sentiment Score](examples/fomc_sentiment_score.png)
*Continuous sentiment scores showing monetary policy evolution and confidence levels*

**Key Insights from Visualization:**
- **2014-2015 Uncertainty:** Negative sentiment during taper tantrum and rate liftoff debates (scores 0.3-0.6)
- **2019-2020 Dovish Pivot:** Shift to positive sentiment as Fed cuts rates and responds to pandemic (scores 0.9+)
- **COVID Response Period:** Mixed sentiment in early 2020 reflecting emergency policy actions and uncertainty
- **2021-2022 Hawkish Turn:** Cluster of negative sentiment in 2020-2021 as inflation concerns emerge (scores 0.8-0.95 showing cautious language)
- **2022 Tightening Cycle:** Single negative sentiment point in late 2022 during aggressive rate hikes
- **Recent Normalization:** Return to predominantly neutral/positive sentiment 2023-2024 (scores 0.7-0.97) as policy stabilizes
- **High Confidence Scores:** Most predictions show 0.7+ confidence, indicating the FinBERT-FOMC model is well-calibrated for Fed language

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
| **Anomaly detection** | COVID-19 labor market shock identification | Detecting unusual workforce patterns, mass layoffs |
| **Temporal trend analysis** | 10-year policy evolution tracking | Long-term workforce demographic shifts |

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
- FinBERT-FOMC (Federal Reserve-specific language model)
- BLS API (Bureau of Labor Statistics)
- SEC EDGAR API

---
