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

# ── Cities — AUTO-DISCOVERED from data/raw/ ───────────────────────────────────
# Any folder placed inside data/raw/ that contains a listings.csv is treated
# as a city dataset automatically.  Add a new city by simply dropping its folder
# into data/raw/ and re-running preprocessing — no code changes needed.

def discover_cities() -> list[str]:
    """Return sorted list of city names found in data/raw/ or fallback to processed data."""
    if RAW_DIR.exists():
        raw_cities = sorted(
            d.name for d in RAW_DIR.iterdir()
            if d.is_dir() and (d / "listings.csv").exists()
        )
        if raw_cities:
            return raw_cities

    # Fallback to processed listings if raw data is not present (e.g. in cloud deployment)
    clean_path = PROCESSED_DIR / "listings_clean.parquet"
    if clean_path.exists():
        try:
            import pandas as pd
            df = pd.read_parquet(clean_path, columns=["city"])
            cities = sorted(df["city"].dropna().unique().tolist())
            if cities:
                return cities
        except Exception:
            pass

    return ["Albany", "Amsterdam", "Antwerp", "Asheville", "Athens"]


# Evaluated lazily so imports work before data/raw/ exists
CITIES: list[str] = discover_cities()

# ── Known city-centre coordinates (lat, lon) ──────────────────────────────────
# This covers all major Airbnb Inside-Airbnb cities.  If a city is not listed
# here its centroid is computed from listing lat/lon at preprocessing time and
# written to data/processed/city_centres.json automatically.
_KNOWN_CITY_CENTRES: dict[str, tuple[float, float]] = {
    # North America
    "Albany":        (42.6526, -73.7562),
    "Asheville":     (35.5951, -82.5515),
    "Austin":        (30.2672, -97.7431),
    "Boston":        (42.3601, -71.0589),
    "Broward County": (26.1901, -80.3659),
    "Cambridge":     (42.3736, -71.1097),
    "Chicago":       (41.8781, -87.6298),
    "Clark County":  (36.1716, -115.1391),
    "Columbus":      (39.9612, -82.9988),
    "Dallas":        (32.7767, -96.7970),
    "Denver":        (39.7392, -104.9903),
    "Detroit":       (42.3314, -83.0458),
    "Hawaii":        (21.3069, -157.8583),
    "Jersey City":   (40.7178, -74.0431),
    "Los Angeles":   (34.0522, -118.2437),
    "Nashville":     (36.1627, -86.7816),
    "New Orleans":   (29.9511, -90.0715),
    "New York City": (40.7128, -74.0060),
    "Newark":        (40.7357, -74.1724),
    "Oakland":       (37.8044, -122.2712),
    "Pacific Grove": (36.6177, -121.9166),
    "Portland":      (45.5231, -122.6765),
    "Rhode Island":  (41.5800, -71.4774),
    "Salem":         (44.9429, -123.0351),
    "San Diego":     (32.7157, -117.1611),
    "San Francisco": (37.7749, -122.4194),
    "San Jose":      (37.3382, -121.8863),
    "San Mateo County": (37.5630, -122.3255),
    "Santa Clara County": (37.3541, -121.9552),
    "Santa Cruz County": (36.9741, -122.0308),
    "Seattle":       (47.6062, -122.3321),
    "Twin Cities":   (44.9778, -93.2650),
    "Washington":    (38.9072, -77.0369),
    # Europe
    "Amsterdam":     (52.3676,   4.9041),
    "Antwerp":       (51.2194,   4.4025),
    "Athens":        (37.9838,  23.7275),
    "Barcelona":     (41.3851,   2.1734),
    "Berlin":        (52.5200,  13.4050),
    "Brussels":      (50.8503,   4.3517),
    "Copenhagen":    (55.6761,  12.5683),
    "Dublin":        (53.3498,  -6.2603),
    "Edinburgh":     (55.9533,  -3.1883),
    "Florence":      (43.7696,  11.2558),
    "Geneva":        (46.2044,   6.1432),
    "Ghent":         (51.0543,   3.7174),
    "Girona":        (41.9794,   2.8214),
    "Hamburg":       (53.5753,  10.0153),
    "Istanbul":      (41.0082,  28.9784),
    "Lisbon":        (38.7169,  -9.1399),
    "London":        (51.5074,  -0.1278),
    "Lyon":          (45.7640,   4.8357),
    "Madrid":        (40.4168,  -3.7038),
    "Mallorca":      (39.6953,   3.0176),
    "Malta":         (35.8997,  14.5147),
    "Menorca":       (39.9498,   4.1112),
    "Milan":         (45.4654,   9.1859),
    "Munich":        (48.1351,  11.5820),
    "Naples":        (40.8518,  14.2681),
    "Oslo":          (59.9139,  10.7522),
    "Paris":         (48.8566,   2.3522),
    "Porto":         (41.1579,  -8.6291),
    "Prague":        (50.0755,  14.4378),
    "Rome":          (41.9028,  12.4964),
    "Seville":       (37.3891,  -5.9845),
    "Stockholm":     (59.3293,  18.0686),
    "Thessaloniki":  (40.6401,  22.9444),
    "Valencia":      (39.4699,  -0.3763),
    "Venice":        (45.4408,  12.3155),
    "Vienna":        (48.2082,  16.3738),
    "Zurich":        (47.3769,   8.5417),
    # Asia-Pacific
    "Bangkok":       (13.7563, 100.5018),
    "Beijing":       (39.9042, 116.4074),
    "Bali":          (-8.3405, 115.0920),
    "Hong Kong":     (22.3193, 114.1694),
    "Melbourne":     (-37.8136, 144.9631),
    "Seoul":         (37.5665, 126.9780),
    "Shanghai":      (31.2304, 121.4737),
    "Singapore":     (1.3521,  103.8198),
    "Sydney":        (-33.8688, 151.2093),
    "Tokyo":         (35.6762, 139.6503),
    # Other
    "Buenos Aires":  (-34.6037, -58.3816),
    "Cape Town":     (-33.9249,  18.4241),
    "Mexico City":   (19.4326, -99.1332),
    "Rio de Janeiro": (-22.9068, -43.1729),
}

# Runtime city-centre lookup (also loads computed centroids if available)
import json as _json

def get_city_centre(city: str) -> tuple[float, float] | None:
    """Return (lat, lon) for a city, checking known dict then computed file."""
    # 1. Exact match
    if city in _KNOWN_CITY_CENTRES:
        return _KNOWN_CITY_CENTRES[city]
    # 2. Case-insensitive match
    city_lower = city.lower()
    for k, v in _KNOWN_CITY_CENTRES.items():
        if k.lower() == city_lower:
            return v
    # 3. Computed centroids file (generated by preprocessing)
    computed_path = PROCESSED_DIR / "city_centres.json"
    if computed_path.exists():
        try:
            with open(computed_path) as f:
                computed = _json.load(f)
            if city in computed:
                return tuple(computed[city])
        except Exception:
            pass
    return None


# Convenience alias — CITY_CENTRES still works for backward compatibility
CITY_CENTRES: dict[str, tuple[float, float]] = _KNOWN_CITY_CENTRES

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
