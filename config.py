"""
config.py — Single source of truth for all project configuration.

All paths are derived from this file's location so the project works
regardless of where it is cloned. Never hardcode absolute paths elsewhere.
"""

from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.resolve()

# ── Data directories ──────────────────────────────────────────────────────────
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# ── Model directories ─────────────────────────────────────────────────────────
MODELS_DIR = ROOT_DIR / "models"
EMBEDDINGS_DIR = MODELS_DIR / "embeddings"
RANKER_DIR = MODELS_DIR / "ranker"

# ── Results ───────────────────────────────────────────────────────────────────
RESULTS_DIR = ROOT_DIR / "results"

# ── Cities ────────────────────────────────────────────────────────────────────
CITIES = ["Albany", "Amsterdam", "Antwerp", "Asheville", "Athens"]

# Approximate city-centre coordinates (lat, lon) — used for geographic features
CITY_CENTRES = {
    "Albany":    (42.6526, -73.7562),
    "Amsterdam": (52.3676,   4.9041),
    "Antwerp":   (51.2194,   4.4025),
    "Asheville": (35.5951, -82.5515),
    "Athens":    (37.9838,  23.7275),
}

# ── Processed file paths ──────────────────────────────────────────────────────
LISTINGS_CLEAN_PATH      = PROCESSED_DIR / "listings_clean.parquet"
REVIEWS_CLEAN_PATH       = PROCESSED_DIR / "reviews_clean.parquet"
CALENDAR_FEATURES_PATH   = PROCESSED_DIR / "calendar_features.parquet"
NEIGHBOURHOODS_PATH      = PROCESSED_DIR / "neighbourhoods_combined.csv"

# ── Embedding / index paths ───────────────────────────────────────────────────
FAISS_INDEX_PATH         = EMBEDDINGS_DIR / "faiss_index.bin"
LISTING_IDS_PATH         = EMBEDDINGS_DIR / "listing_ids.npy"
EMBEDDINGS_MATRIX_PATH   = EMBEDDINGS_DIR / "listing_embeddings.npy"
TFIDF_VECTORIZER_PATH    = EMBEDDINGS_DIR / "tfidf_vectorizer.pkl"
TFIDF_MATRIX_PATH        = EMBEDDINGS_DIR / "tfidf_matrix.npz"

# ── Ranker paths ──────────────────────────────────────────────────────────────
XGB_RANKER_PATH          = RANKER_DIR / "xgb_ranker.json"
FEATURE_NAMES_PATH       = RANKER_DIR / "feature_names.json"
TRAIN_FEATURES_PATH      = RANKER_DIR / "train_features.parquet"
TEST_FEATURES_PATH       = RANKER_DIR / "test_features.parquet"

# ── Results paths ─────────────────────────────────────────────────────────────
EVAL_RESULTS_PATH        = RESULTS_DIR / "evaluation_results.json"
QUERIES_PATH             = RESULTS_DIR / "synthetic_queries.json"

# ── Sentence Transformer model ────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM        = 384   # dimension for all-MiniLM-L6-v2

# ── Search hyperparameters ────────────────────────────────────────────────────
HYBRID_ALPHA    = 0.6   # weight for semantic score; (1-α) for TF-IDF
TOP_K_RETRIEVE  = 150   # candidates retrieved per search before ranking
TOP_K_RESULTS   = 10    # final results returned to user
TFIDF_MAX_FEATURES = 50_000
TFIDF_NGRAM_RANGE  = (1, 2)

# ── Ranking model hyperparameters ─────────────────────────────────────────────
XGB_PARAMS = {
    "objective":        "rank:ndcg",
    "n_estimators":     300,
    "learning_rate":    0.05,
    "max_depth":        6,
    "min_child_weight": 5,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "random_state":     42,
    "n_jobs":           -1,
    "tree_method":      "hist",   # fast CPU training
    "eval_metric":      "ndcg@10",
}

# ── Synthetic label weights ───────────────────────────────────────────────────
LABEL_WEIGHTS = {
    "semantic_sim":         0.35,
    "price_match":          0.20,
    "amenity_match_ratio":  0.20,
    "rating_norm":          0.15,
    "review_count_norm":    0.10,
}

# ── Query generation ──────────────────────────────────────────────────────────
NUM_QUERIES      = 1500   # synthetic queries for training + evaluation
TRAIN_QUERY_FRAC = 0.80   # 80 % train / 20 % test (query-level split)

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Preprocessing thresholds ─────────────────────────────────────────────────
PRICE_MIN      = 5      # drop listings cheaper than $5/night
PRICE_MAX      = 10_000 # drop obvious outliers
MIN_LISTING_TEXT_LEN = 10  # drop listings with virtually no text

# ── Calendar processing ───────────────────────────────────────────────────────
CALENDAR_CHUNKSIZE = 100_000  # rows per chunk when reading large calendar files

# ── Top amenities to track as binary features ─────────────────────────────────
TOP_AMENITIES = [
    "Wifi", "Kitchen", "Air conditioning", "Heating", "Washer",
    "Dryer", "Free parking on premises", "TV", "Dishwasher",
    "Elevator", "Workspace", "Hair dryer", "Iron", "Coffee maker",
    "Microwave", "Refrigerator", "Oven", "Hot water", "Shampoo",
    "Long term stays allowed", "Smoke alarm", "Carbon monoxide alarm",
    "Fire extinguisher", "First aid kit", "Pool", "Gym",
    "Breakfast", "Crib", "High chair", "Self check-in",
]


def ensure_dirs() -> None:
    """Create all required directories if they don't exist."""
    for d in [PROCESSED_DIR, EMBEDDINGS_DIR, RANKER_DIR, RESULTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
