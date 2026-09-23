# Intelligent Airbnb Search & Ranking Engine

> A production-oriented multi-city accommodation search and ranking prototype implementing core components of a modern marketplace search system using publicly available [Inside Airbnb](https://insideairbnb.com/) data.

---

## Overview

This system allows a user to enter a natural-language query such as:

> *"Quiet apartment near the city centre in Amsterdam with WiFi and kitchen under $150"*

…and receive ranked, explained accommodation listings powered by:
- **Hybrid retrieval** — TF-IDF keyword search + Sentence Transformer semantic search
- **ML Ranking** — XGBoost ranker trained on query-listing features
- **Explainable results** — per-listing explanations grounded in actual feature values

**Dataset**: 5 cities — Albany, Amsterdam, Antwerp, Asheville, Athens (~30,597 listings)  
**Budget**: ₹0 / $0 — fully free and open-source  
**Hardware**: Local CPU only — no GPU required

---

## Architecture

```
User Query (natural language)
        │
        ▼
  Query Processing
        │
   ┌────┴────┐
   ▼         ▼
TF-IDF    Sentence Transformer
Keyword    Semantic Search
Search     (all-MiniLM-L6-v2)
   │         │
   └────┬────┘
        ▼
  Hybrid Retrieval
  (α·semantic + (1-α)·keyword)
        │
        ▼
  Candidate Set (~150 listings)
        │
        ▼
  Hard Filters
  (price, city, room type, guests)
        │
        ▼
  Feature Engineering
        │
        ▼
  XGBoost Ranker
        │
        ▼
  Top-10 Results + Explanations
        │
        ▼
  Streamlit UI
```

---

## Dataset

**Source**: [Inside Airbnb](https://insideairbnb.com/get-the-data/) — publicly available data  
**Cities covered**: Albany (NY), Amsterdam, Antwerp, Asheville (NC), Athens  
**Files used**: `listings.csv`, `reviews.csv`, `calendar.csv`, `neighbourhoods.csv`, `neighbourhoods.geojson`

> **Disclaimer**: This project uses independently collected publicly available data. It is not affiliated with, endorsed by, or representative of Airbnb Inc. Synthetic relevance labels are used — this is **not** real Airbnb user behaviour data.

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/airbnb-ranking-engine.git
cd airbnb-ranking-engine

# 2. Create a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows PowerShell
# source .venv/bin/activate   # Linux/macOS

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

```bash
# Step 1 — Preprocess all city data
python src/preprocessing.py

# Step 2 — Build search index (TF-IDF + FAISS embeddings)
python src/search.py --build-index

# Step 3 — Train ranking model
python src/ranking.py --train

# Step 4 — Evaluate all models
python evaluation/evaluate.py

# Step 5 — Launch the Streamlit app
streamlit run app.py
```

---

## Project Structure

```
airbnb_ranking_engine/
├── data/
│   ├── raw/                  # Place city folders here (not committed to git)
│   └── processed/            # Generated parquet files
├── models/
│   ├── embeddings/           # FAISS index + TF-IDF vectorizer
│   └── ranker/               # Trained XGBoost model
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_semantic_search.ipynb
│   └── 04_ranking_model.ipynb
├── src/
│   ├── preprocessing.py      # Data cleaning pipeline
│   ├── search.py             # TF-IDF + FAISS + hybrid retrieval
│   ├── features.py           # Feature engineering
│   └── ranking.py            # XGBoost ranker training
├── evaluation/
│   └── evaluate.py           # Retrieval & ranking metrics
├── results/                  # Evaluation outputs
├── app.py                    # Streamlit UI
├── api.py                    # FastAPI backend
├── config.py                 # All configuration
├── requirements.txt
└── Dockerfile
```

---

## Evaluation

> Metrics will be filled in after running `python evaluation/evaluate.py`

| Approach | Recall@100 | MRR@10 | NDCG@10 | P95 Latency |
|---|---|---|---|---|
| TF-IDF | — | — | — | — |
| Semantic Search | — | — | — | — |
| Hybrid Search | — | — | — | — |
| Hybrid + XGBoost | — | — | — | — |

---

## Limitations

- **Synthetic relevance labels**: No real Airbnb click/booking data is available. Labels are derived from semantic similarity, price match, amenity match, and rating signals.
- **Antwerp data**: Only 19 columns available (no descriptions, amenities, or rating breakdown). Antwerp listings have shorter text representations.
- **Static data**: The dataset is a snapshot — prices and availability are not real-time.
- **English-centric**: The embedding model is optimised for English; non-English descriptions may have lower retrieval quality.

---

## Future Work

- BM25 upgrade from TF-IDF baseline
- Cross-encoder reranking
- Review sentiment / aspect extraction
- Multilingual query support
- Real-time availability integration
- User preference personalisation

---

## License / Attribution

Dataset: Inside Airbnb ([insideairbnb.com](https://insideairbnb.com)) — data is made available under a [Creative Commons CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/) licence.  
This project is independent and not affiliated with Airbnb Inc.
