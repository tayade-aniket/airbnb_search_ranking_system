"""
src/ranking.py
==============
Synthetic query generation, relevance labelling, and XGBoost ranker training.

Pipeline
--------
1. Generate ~1,500 representative search queries across 5 cities.
2. For each query, retrieve hybrid candidates via AirbnbSearchEngine.
3. Build feature matrix using FeatureEngineer.
4. Compute synthetic relevance labels (explicitly NOT real Airbnb user data).
5. Train XGBRanker with rank:ndcg objective.
6. Save model and feature names.

Usage
-----
    # Full training pipeline
    python src/ranking.py --train

    # Only generate queries (no training)
    python src/ranking.py --generate-queries

    # Score a single query against loaded ranker
    python src/ranking.py --score "quiet apartment in Amsterdam with WiFi under $150"
"""

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.features import FeatureEngineer, FEATURE_NAMES, parse_query_intent

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  SYNTHETIC QUERY GENERATION
# ─────────────────────────────────────────────────────────────────────────────

# Query templates keyed by intent category
_QUERY_TEMPLATES = {
    "price": [
        "cheap apartment in {city}",
        "budget stay in {city} under ${price}",
        "affordable accommodation in {city}",
        "inexpensive room in {city} under ${price} per night",
        "cheap place to stay in {city}",
        "economy apartment in {city}",
    ],
    "location": [
        "apartment near {landmark} in {city}",
        "place near city centre in {city}",
        "accommodation near public transport in {city}",
        "flat close to downtown {city}",
        "stay near {landmark}",
        "room near the centre of {city}",
    ],
    "amenity": [
        "apartment with WiFi in {city}",
        "place with kitchen in {city}",
        "accommodation with workspace in {city}",
        "apartment with WiFi and kitchen in {city}",
        "place with free parking in {city}",
        "apartment with air conditioning in {city}",
        "room with washer in {city}",
        "place with pool in {city}",
    ],
    "intent": [
        "quiet apartment in {city}",
        "family-friendly apartment in {city}",
        "romantic stay in {city}",
        "business accommodation in {city}",
        "luxury apartment in {city}",
        "cosy studio in {city}",
        "pet-friendly place in {city}",
    ],
    "combined": [
        "quiet apartment in {city} with WiFi under ${price}",
        "entire apartment in {city} with kitchen and WiFi under ${price}",
        "private room in {city} near city centre under ${price}",
        "family apartment in {city} with WiFi and kitchen under ${price}",
        "luxury apartment in {city} with pool",
        "cosy apartment near {landmark} in {city} with workspace",
        "cheap entire home in {city} for 2 guests under ${price}",
        "business stay in {city} with WiFi and workspace under ${price}",
    ],
}

_CITY_LANDMARKS = {
    "Albany":    ["the Capitol", "Washington Park", "downtown Albany"],
    "Amsterdam": ["the Rijksmuseum", "Dam Square", "Vondelpark", "the Anne Frank House"],
    "Antwerp":   ["the Cathedral", "Grote Markt", "the port area"],
    "Asheville": ["the Biltmore Estate", "downtown Asheville", "River Arts District"],
    "Athens":    ["the Acropolis", "Monastiraki", "Syntagma Square", "Plaka"],
}

_PRICE_OPTIONS = [50, 75, 100, 125, 150, 175, 200, 250, 300]


def generate_queries(n: int = config.NUM_QUERIES, seed: int = config.RANDOM_SEED) -> list[dict]:
    """
    Generate n synthetic search queries across 5 cities and 5 intent categories.

    Returns a list of dicts:
        {
            "query_id": str,
            "query": str,
            "city": str | None,
            "max_price": float | None,
            "category": str,
        }

    IMPORTANT: These are synthetic/simulated queries.
    They do NOT represent real Airbnb user behaviour.
    """
    rng = random.Random(seed)
    queries = []
    categories = list(_QUERY_TEMPLATES.keys())
    per_category = n // len(categories)
    available_cities = config.discover_cities() or config.CITIES or ["Amsterdam"]

    for cat in categories:
        templates = _QUERY_TEMPLATES[cat]
        for i in range(per_category):
            city = rng.choice(available_cities)
            default_landmarks = ["the city centre", "the central station", "downtown", "the old town"]
            landmark = rng.choice(_CITY_LANDMARKS.get(city, default_landmarks))
            price = rng.choice(_PRICE_OPTIONS)
            template = rng.choice(templates)

            query = (
                template
                .replace("{city}", city)
                .replace("{landmark}", landmark)
                .replace("{price}", str(price))
            )

            queries.append({
                "query_id": f"{cat}_{i:04d}",
                "query": query,
                "city": city,
                "max_price": price if "{price}" in template else None,
                "category": cat,
            })

    rng.shuffle(queries)
    log.info("Generated %d synthetic queries across %d categories for %d cities.", len(queries), len(categories), len(available_cities))
    return queries


# ─────────────────────────────────────────────────────────────────────────────
# 2.  SYNTHETIC RELEVANCE LABELLING
# ─────────────────────────────────────────────────────────────────────────────

def compute_relevance_label(
    features: pd.Series,
    weights: dict = config.LABEL_WEIGHTS,
) -> int:
    """
    Compute a synthetic relevance grade 0–3 for one (query, listing) pair.

    Formula (from Master Prompt):
        relevance = 0.35 × semantic_sim
                  + 0.20 × price_match
                  + 0.20 × amenity_match_ratio
                  + 0.15 × rating_norm
                  + 0.10 × review_count_log_norm

    Grades:
        0 = irrelevant   (< 0.25)
        1 = somewhat     (0.25–0.50)
        2 = relevant     (0.50–0.70)
        3 = highly       (>= 0.70)

    SYNTHETIC DATA DISCLAIMER:
        These labels are derived from computed feature values, NOT from real
        Airbnb user click or booking data.
    """
    score = (
        weights.get("semantic_sim", 0.35)      * float(features.get("semantic_score", 0))
        + weights.get("price_match", 0.20)     * float(features.get("price_match", 0.5))
        + weights.get("amenity_match_ratio", 0.20) * float(features.get("amenity_match", 0.5))
        + weights.get("rating_norm", 0.15)     * float(features.get("rating_norm", 0))
        + weights.get("review_count_norm", 0.10) * min(float(features.get("reviews_log", 0)) / 8.0, 1.0)
    )

    if score >= 0.70:
        return 3
    elif score >= 0.50:
        return 2
    elif score >= 0.25:
        return 1
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# 3.  DATASET BUILDING
# ─────────────────────────────────────────────────────────────────────────────

def build_training_dataset(
    queries: list[dict],
    engine,
    fe: FeatureEngineer,
) -> pd.DataFrame:
    """
    For each query:
      1. Retrieve top-150 hybrid candidates (city-filtered if city in query).
      2. Build feature matrix.
      3. Compute synthetic relevance labels.

    Returns a DataFrame with columns = ['query_id', 'listing_id', 'relevance'] + FEATURE_NAMES
    """
    all_rows = []
    skipped = 0

    for i, q_info in enumerate(queries):
        query     = q_info["query"]
        city      = q_info.get("city")
        max_price = q_info.get("max_price")
        qid       = q_info["query_id"]

        city_filter = [city] if city else None

        try:
            candidates = engine.hybrid_search(
                query,
                top_k=config.TOP_K_RETRIEVE,
                city_filter=city_filter,
                max_price=max_price,
            )
        except Exception as e:
            log.warning("Query %s failed retrieval: %s", qid, e)
            skipped += 1
            continue

        if candidates.empty:
            skipped += 1
            continue

        # Build features
        feat_df = fe.build_features(query, candidates, max_price=max_price)

        if feat_df.empty:
            skipped += 1
            continue

        # Add relevance labels
        feat_df["relevance"] = feat_df.apply(compute_relevance_label, axis=1)
        feat_df["query_id"]  = qid

        all_rows.append(feat_df)

        if (i + 1) % 100 == 0:
            log.info("  Processed %d/%d queries…", i + 1, len(queries))

    if skipped:
        log.warning("Skipped %d queries (empty candidates or error).", skipped)

    if not all_rows:
        raise RuntimeError("No training data generated — check search engine is loaded.")

    dataset = pd.concat(all_rows, ignore_index=True)
    log.info("Training dataset: %d (query, listing) pairs from %d queries",
             len(dataset), len(all_rows))
    return dataset


# ─────────────────────────────────────────────────────────────────────────────
# 4.  TRAIN / TEST SPLIT (query-level)
# ─────────────────────────────────────────────────────────────────────────────

def query_level_split(
    dataset: pd.DataFrame,
    train_frac: float = config.TRAIN_QUERY_FRAC,
    seed: int = config.RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split at the query level (not listing level) to prevent data leakage.
    Queries are shuffled and split 80/20.
    """
    query_ids = dataset["query_id"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(query_ids)
    n_train = int(len(query_ids) * train_frac)
    train_qids = set(query_ids[:n_train])
    test_qids  = set(query_ids[n_train:])

    train = dataset[dataset["query_id"].isin(train_qids)].copy()
    test  = dataset[dataset["query_id"].isin(test_qids)].copy()

    log.info("Train: %d pairs from %d queries | Test: %d pairs from %d queries",
             len(train), len(train_qids), len(test), len(test_qids))
    return train, test


# ─────────────────────────────────────────────────────────────────────────────
# 5.  XGBOOST RANKER TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_ranker(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> "xgb.XGBRanker":
    """
    Train an XGBRanker using rank:ndcg objective.

    XGBoost expects:
      - X: feature matrix (float)
      - y: relevance labels (int)
      - group: array of group sizes (one per query)
    """
    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError("xgboost is not installed. Run: pip install xgboost")

    def _prepare(df: pd.DataFrame):
        # Ensure all feature columns exist; fill missing with 0
        feat_cols = [c for c in FEATURE_NAMES if c in df.columns]
        missing   = [c for c in FEATURE_NAMES if c not in df.columns]
        if missing:
            log.warning("Missing feature columns (will be filled with 0): %s", missing)
            for c in missing:
                df[c] = 0.0

        X = df[FEATURE_NAMES].values.astype(np.float32)
        y = df["relevance"].values.astype(np.int32)

        # group sizes: number of listings per query, in query order
        groups = df.groupby("query_id", sort=False).size().values
        return X, y, groups

    X_train, y_train, g_train = _prepare(train)
    X_test,  y_test,  g_test  = _prepare(test)

    log.info("Training XGBRanker: %d train samples, %d test samples…", len(X_train), len(X_test))
    log.info("Feature count: %d | Label distribution: %s",
             X_train.shape[1],
             dict(zip(*np.unique(y_train, return_counts=True))))

    ranker = xgb.XGBRanker(**config.XGB_PARAMS)
    ranker.fit(
        X_train, y_train,
        group=g_train,
        eval_set=[(X_test, y_test)],
        eval_group=[g_test],
        verbose=50,
    )

    log.info("Training complete.")
    return ranker


# ─────────────────────────────────────────────────────────────────────────────
# 6.  RANKING INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

class XGBRankingModel:
    """
    Wrapper for loading and applying the trained XGBoost ranker.

    Example
    -------
    model = XGBRankingModel()
    model.load()
    ranked_df = model.rank(query, candidates_df, feature_engineer, max_price=150)
    """

    def __init__(self):
        self._model = None
        self._fe: Optional[FeatureEngineer] = None
        self._loaded = False

    def load(self, fe: Optional[FeatureEngineer] = None) -> None:
        """Load the trained XGBoost model from disk."""
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("xgboost is not installed. Run: pip install xgboost")

        if not config.XGB_RANKER_PATH.exists():
            raise FileNotFoundError(
                f"Ranker not found at {config.XGB_RANKER_PATH}. "
                "Run: python src/ranking.py --train"
            )

        self._model = xgb.XGBRanker()
        self._model.load_model(str(config.XGB_RANKER_PATH))
        self._fe = fe
        self._loaded = True
        log.info("XGBRanker loaded from %s", config.XGB_RANKER_PATH)

    def rank(
        self,
        query: str,
        candidates: pd.DataFrame,
        fe: Optional[FeatureEngineer] = None,
        max_price: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Re-rank a candidate DataFrame using the XGBoost model.

        Returns candidates sorted by xgb_score descending.
        """
        if not self._loaded:
            raise RuntimeError("Call .load() before .rank()")

        feature_engineer = fe or self._fe
        if feature_engineer is None:
            raise ValueError("A FeatureEngineer instance is required.")

        feat_df = feature_engineer.build_features(query, candidates, max_price=max_price)
        if feat_df.empty:
            return candidates

        # Fill any missing feature columns
        for col in FEATURE_NAMES:
            if col not in feat_df.columns:
                feat_df[col] = 0.0

        X = feat_df[FEATURE_NAMES].values.astype(np.float32)
        scores = self._model.predict(X)

        feat_df["xgb_score"] = scores
        feat_df = feat_df.sort_values("xgb_score", ascending=False)

        # Merge scores back onto candidates
        candidates = candidates.copy()
        score_map = dict(zip(feat_df["listing_id"], feat_df["xgb_score"]))
        feat_map  = feat_df.set_index("listing_id").to_dict("index")

        candidates["xgb_score"] = candidates["id"].map(score_map).fillna(0.0)
        # Attach feature values for explanation
        for feat_col in ["semantic_score", "tfidf_score", "amenity_match",
                         "price_match", "rating_norm", "reviews_log",
                         "superhost", "dist_city_centre_km"]:
            candidates[f"feat_{feat_col}"] = candidates["id"].map(
                lambda lid, fc=feat_col: feat_map.get(lid, {}).get(fc, 0.0)
            )

        return candidates.sort_values("xgb_score", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  SAVE / LOAD HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def save_ranker(ranker, feature_names: list) -> None:
    """Save XGBoost model and feature names to disk."""
    config.ensure_dirs()
    ranker.save_model(str(config.XGB_RANKER_PATH))
    with open(config.FEATURE_NAMES_PATH, "w") as f:
        json.dump(feature_names, f)
    log.info("Ranker saved to %s", config.XGB_RANKER_PATH)


def save_queries(queries: list[dict]) -> None:
    """Save synthetic queries to disk for reproducibility."""
    config.ensure_dirs()
    with open(config.QUERIES_PATH, "w") as f:
        json.dump(queries, f, indent=2)
    log.info("Queries saved to %s", config.QUERIES_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run_training() -> None:
    """Full training pipeline."""
    from src.search import AirbnbSearchEngine

    log.info("╔══════════════════════════════════════════════════╗")
    log.info("║      Airbnb Ranking Engine — Ranker Training     ║")
    log.info("╚══════════════════════════════════════════════════╝")

    # Load search engine
    engine = AirbnbSearchEngine()
    engine.load()

    # Feature engineer
    fe = FeatureEngineer(engine.listings)

    # Generate queries
    log.info("Generating %d synthetic queries…", config.NUM_QUERIES)
    queries = generate_queries(n=config.NUM_QUERIES)
    save_queries(queries)

    # Build dataset
    log.info("Building training dataset (retrieval + feature engineering)…")
    t0 = time.perf_counter()
    dataset = build_training_dataset(queries, engine, fe)
    log.info("Dataset built in %.1f s", time.perf_counter() - t0)

    # Split
    train, test = query_level_split(dataset)

    # Save for evaluation reuse
    config.ensure_dirs()
    train.to_parquet(config.TRAIN_FEATURES_PATH, index=False)
    test.to_parquet(config.TEST_FEATURES_PATH, index=False)

    # Train
    ranker = train_ranker(train, test)

    # Save
    save_ranker(ranker, FEATURE_NAMES)
    log.info("Training pipeline complete ✓")


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Airbnb XGBoost Ranker")
    parser.add_argument("--train", action="store_true", help="Run full training pipeline")
    parser.add_argument("--generate-queries", action="store_true",
                        help="Generate and save synthetic queries only")
    parser.add_argument("--score", type=str, default=None,
                        help="Score a query using the loaded ranker (requires index built)")
    args = parser.parse_args()

    if args.generate_queries:
        queries = generate_queries()
        save_queries(queries)
        print(f"Generated {len(queries)} queries → {config.QUERIES_PATH}")
        for q in queries[:5]:
            print(f"  [{q['category']}] {q['query']}")

    if args.train:
        run_training()

    if args.score:
        from src.search import AirbnbSearchEngine
        engine = AirbnbSearchEngine()
        engine.load()
        fe = FeatureEngineer(engine.listings)
        model = XGBRankingModel()
        model.load(fe)
        candidates = engine.hybrid_search(args.score, top_k=20)
        ranked = model.rank(args.score, candidates, fe)
        cols = ["id", "city", "name", "price_usd", "xgb_score",
                "review_scores_rating", "room_type"]
        print(ranked[[c for c in cols if c in ranked.columns]].head(10).to_string(index=False))


if __name__ == "__main__":
    _cli()
