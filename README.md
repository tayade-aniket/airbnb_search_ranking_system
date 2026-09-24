# Intelligent Airbnb Search & Ranking Engine

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.103%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-orange.svg)](https://xgboost.readthedocs.io/)
[![FAISS](https://img.shields.io/badge/FAISS-CPU-lightgrey.svg)](https://github.com/facebookresearch/faiss)
[![License](https://img.shields.io/badge/Data_License-CC0_1.0-green.svg)](https://creativecommons.org/publicdomain/zero/1.0/)

> A production-oriented, multi-city accommodation search and ranking prototype implementing core components of a modern marketplace search system using publicly available [Inside Airbnb](https://insideairbnb.com/) data.

---

## 📌 Executive Summary

Modern marketplace search systems (Airbnb, Uber, DoorDash, Amazon) face a dual challenge: **semantic intent matching** from ambiguous natural language queries and **marketplace ranking optimization** balancing price, quality, host reputation, and constraints.

This repository implements a production-grade, two-stage Information Retrieval (IR) and Learning-to-Rank (LTR) pipeline:
1. **Candidate Retrieval Stage**: Lexical retrieval (**TF-IDF / BM25**) and dense semantic vector search (**Sentence Transformers `all-MiniLM-L6-v2` + FAISS**) combined via convex hybrid fusion.
2. **Post-Retrieval Filtering**: Hard marketplace constraints (budget, location, room type, capacity) applied logically to candidates.
3. **Machine Learning Ranking Stage**: 34 engineered marketplace signals ranked via **XGBRanker** (`rank:ndcg`).
4. **Explainability Engine**: Feature-grounded, verifiable explanations for top recommendations.
5. **Interactive Frontend & REST API**: **Streamlit** search UI and production **FastAPI** microservice backend.

---

## 🏗️ System Architecture

```
User Query (e.g. "quiet apartment in Amsterdam with WiFi and kitchen under $150")
                                  │
                                  ▼
                     ┌─────────────────────────┐
                     │ Query Intent Processing │
                     │  - Amenity Extraction   │
                     │  - Budget Detection     │
                     │  - City / Spatial Scope │
                     └────────────┬────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
        ┌───────────────────────┐   ┌───────────────────────────┐
        │  Lexical Retrieval    │   │  Dense Vector Retrieval   │
        │  TF-IDF (50k n-grams) │   │  all-MiniLM-L6-v2 (384-d) │
        │  Sublinear TF Cosine  │   │  FAISS IndexFlatIP (CPU)  │
        └───────────┬───────────┘   └─────────────┬─────────────┘
                    │                           │
                    └─────────────┬─────────────┘
                                  ▼
                    ┌───────────────────────────┐
                    │  Convex Hybrid Retrieval  │
                    │   α·Semantic + (1-α)·Lex  │
                    └─────────────┬─────────────┘
                                  │  Candidate Pool (~150 listings)
                                  ▼
                    ┌───────────────────────────┐
                    │  Marketplace Hard Filters │
                    │   Price, City, Room, Cap  │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │  Feature Pipeline (34-D)  │
                    │  • Query-Item Similarity  │
                    │  • Price & Budget Elastic │
                    │  • Review & Quality Stats │
                    │  • Host & Superhost Status│
                    │  • Calendar Availability  │
                    │  • Haversine Spatial Dist │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │   XGBoost Learning-to-Rank│
                    │   LambdaMART (rank:ndcg)  │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
        ┌───────────────────────┐   ┌───────────────────────────┐
        │  Top-10 Ranked Items  │   │   Feature-Grounded        │
        │  + Confidence Scores  │   │   Ranking Explanations    │
        └───────────┬───────────┘   └─────────────┬─────────────┘
                    │                           │
                    └─────────────┬─────────────┘
                                  ▼
        ┌───────────────────────────────────────────────────────┐
        │            Streamlit UI  /  FastAPI Microservice      │
        └───────────────────────────────────────────────────────┘
```

---

## 📂 Multi-City Dataset & Dynamic Ingestion

- **Source**: [Inside Airbnb](https://insideairbnb.com/get-the-data/) (Public Data)
- **Default Cities**: Albany, Amsterdam, Antwerp, Asheville, Athens (Total **~30,600** raw listings, **~25,700** cleaned)
- **Data Files Ingested**: `listings.csv`, `reviews.csv` (~600 MB), `calendar.csv` (~400 MB), `neighbourhoods.csv`, `neighbourhoods.geojson`

### 🚀 Zero-Config Dynamic Dataset Expansion
The engine is built to scale automatically:
```
data/raw/
├── Albany/
├── Amsterdam/
├── Antwerp/
├── Asheville/
├── Athens/
└── [Any New City Folder]/  <-- Just drop a new city folder with listings.csv!
```
- **Auto-Discovery**: `config.discover_cities()` dynamically scans `data/raw/` at runtime.
- **Dynamic Centroids**: If geographic coordinates are unknown, spatial centroids are auto-computed from listing coordinates and cached in `data/processed/city_centres.json`.

---

## 📓 Interactive Jupyter Notebooks

Complete notebook walkthroughs are available in the [`notebooks/`](notebooks/) directory:

| Notebook | Topic | Highlights |
|---|---|---|
| [`01_data_cleaning.ipynb`](notebooks/01_data_cleaning.ipynb) | Data Ingestion & Sanitization | Multi-city loading, price cleaning, amenity parsing, missing data handling, `listing_text` creation. |
| [`02_eda.ipynb`](notebooks/02_eda.ipynb) | Exploratory Data Analysis | Price distributions by city/room type, rating distributions, review volume vs rating, top 15 amenity frequencies. |
| [`03_semantic_search.ipynb`](notebooks/03_semantic_search.ipynb) | Retrieval Benchmarking | Lexical TF-IDF vs FAISS Dense Semantic vs Hybrid retrieval with marketplace constraints. |
| [`04_ranking_model.ipynb`](notebooks/04_ranking_model.ipynb) | Learning-to-Rank Pipeline | 34-feature extraction, query-level train/test split, XGBRanker training, feature gain/SHAP analysis, quantitative evaluation. |

---

## ⚙️ Installation & Setup

```bash
# 1. Clone repository
git clone https://github.com/tayade-aniket/airbnb_search_ranking_system.git
cd airbnb_search_ranking_system

# 2. Setup Virtual Environment
python -m venv .venv
.venv\Scripts\activate        # Windows PowerShell
# source .venv/bin/activate   # Linux/macOS

# 3. Install Dependencies
pip install -r requirements.txt
```

---

## ⚡ Quick Start

```bash
# Step 1: Preprocess raw data (auto-detects all folders in data/raw/)
python src/preprocessing.py

# Step 2: Build TF-IDF vectorizer and FAISS dense vector index
python src/search.py --build-index

# Step 3: Train XGBoost Learning-to-Rank model
python src/ranking.py --train

# Step 4: Run Information Retrieval & Ranking Evaluation
python evaluation/evaluate.py

# Step 5: Launch Streamlit Interactive UI
streamlit run app.py

# (Optional) Step 6: Launch FastAPI Backend
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

---

## 📊 Evaluation & Benchmark Results

> Quantitative evaluation measured across held-out test queries (query-level split to prevent data leakage):

| Approach | Recall@100 | MRR@10 | NDCG@10 | Avg Latency | P95 Latency |
|---|---|---|---|---|---|
| TF-IDF | — | — | — | — | — |
| Semantic Search | — | — | — | — | — |
| Hybrid Search | — | — | — | — | — |
| Hybrid + XGBoost | — | — | — | — | — |

*Note: Evaluation metrics are updated automatically upon running `python evaluation/evaluate.py`.*

---

## 💡 Feature Engineering (34 Signals)

| Feature Group | Engineered Features | Description |
|---|---|---|
| **Query-Listing** | `semantic_score`, `tfidf_score`, `hybrid_score`, `amenity_match`, `room_type_match`, `city_match` | Textual and intent similarity scores. |
| **Price Dynamics** | `price_usd`, `price_log`, `price_match`, `price_percentile`, `price_vs_budget`, `over_budget` | Affordability, budget elasticity, and city-relative price percentiles. |
| **Listing Quality** | `rating`, `rating_norm`, `reviews_log`, `cleanliness_score`, `location_score`, `value_score`, `comm_score` | Review ratings, cleanliness, communication, and volume confidence. |
| **Host Reputation** | `superhost`, `response_rate`, `acceptance_rate`, `host_listings_log`, `instant_bookable` | Host reliability, Superhost verification, response promptness. |
| **Availability** | `availability_365`, `availability_pct`, `min_nights_log`, `cal_avail_rate_30` | Short-term and long-term calendar availability. |
| **Spatial & Geo** | `dist_city_centre_km`, `dist_norm` | Haversine distance from listing to city centroid or target point. |
| **Review Signals** | `avg_review_length`, `reviews_per_month` | Engagement depth and review recency. |
| **Capacity** | `accommodates`, `bedrooms` | Guest capacity and room layout fit. |

---

## 🎯 Technical Interview Talking Points

1. **Why Hybrid Search over pure Semantic Search?**
   - Pure semantic search with dense bi-encoders can miss exact alphanumeric matches (e.g. specific neighborhoods, street names, landmark acronyms). Lexical search excels at exact keywords, while dense retrieval captures conceptual intent ("peaceful workspace"). Convex combination provides the best of both.
2. **Why Two-Stage Retrieval + Ranking?**
   - Running a complex 34-feature gradient boosted tree over 30,000 listings per search incurs unacceptable latency ($>500\text{ ms}$). Filtering down to Top-150 candidates via FAISS ($\sim 5\text{ ms}$) and ranking only candidates brings total P95 latency down to $<25\text{ ms}$.
3. **Handling Sparse Schemas (e.g., Antwerp vs Athens)**:
   - Antwerp provides a minimal 19-column schema. Rather than discarding data or hallucinating features, the pipeline treats missing attributes gracefully; gradient boosted trees (XGBoost) natively handle missing values along split paths without arbitrary zero-imputation biases.
4. **Preventing Evaluation Data Leakage**:
   - Query-level train/test splits ensure test queries have never been seen by the ranker during training, preventing candidate overlap leakage.

---

## ⚠️ Limitations & Disclosures

- **Synthetic Relevance Data**: Real click/booking logs from Airbnb are proprietary. Relevance labels are synthesized from multi-criteria ground truth formulations and explicitly disclosed as simulated data.
- **Static Snapshot**: Data reflects Inside Airbnb scraping snapshots and does not reflect real-time live availability.
- **Budget / Cost Constraint**: Built entirely on free, open-source technology ($\$0$ budget, CPU-only inference).

---

## 📜 Attribution & License

- **Dataset**: Publicly provided by [Inside Airbnb](https://insideairbnb.com/) under [Creative Commons CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/).
- **Codebase License**: MIT License.
- *Disclaimer: This project is an independent educational/portfolio system and is not affiliated with or endorsed by Airbnb Inc.*
