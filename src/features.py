"""
src/features.py
===============
Feature engineering for the XGBoost ranking model.

For each (query, listing) pair, computes a feature vector covering:
  - Query-Listing similarity (semantic, keyword, amenity match)
  - Price features (absolute, relative to budget, percentile)
  - Quality features (ratings, review count)
  - Host features (superhost, response rate, acceptance rate)
  - Availability features
  - Geographic features (distance from city centre)
  - Review-derived features (avg length, recency)
  - Calendar-derived features (availability rate)

Usage
-----
    from src.features import FeatureEngineer
    fe = FeatureEngineer(listings_df)
    feature_matrix = fe.build_features(query, candidates_df, max_price=150)
"""

import logging
import math
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance between two (lat, lon) points in km."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1))
         * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def _log_norm(val: float, epsilon: float = 1.0) -> float:
    """log(1 + x) normalised to a positive value; safe for 0 and NaN."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return 0.0
    return math.log1p(max(float(val), 0.0))


def _safe_float(val, default: float = 0.0) -> float:
    try:
        f = float(val)
        return default if math.isnan(f) else f
    except (TypeError, ValueError):
        return default


# ─────────────────────────────────────────────────────────────────────────────
# QUERY PARSING
# ─────────────────────────────────────────────────────────────────────────────

def parse_query_intent(query: str) -> dict:
    """
    Extract structured signals from a natural-language query.
    Returns a dict with:
      - mentioned_amenities : list[str]
      - mentioned_city      : str | None
      - budget_hint         : float | None  (extracted price like "$150")
      - room_type_hint      : str | None    ("entire", "private", "shared")
      - intent_tags         : list[str]     (["quiet","family","luxury","business"])
    """
    q = query.lower()

    # Budget extraction: "under $150", "$200", "less than 180"
    budget_hint = None
    budget_match = re.search(
        r"(?:under|below|less than|max|maximum|budget)?\s*\$?(\d{2,4})\s*(?:per night|/night|usd)?",
        q,
    )
    if budget_match:
        budget_hint = float(budget_match.group(1))

    # City detection
    mentioned_city = None
    for city in config.CITIES:
        if city.lower() in q:
            mentioned_city = city
            break

    # Room type hint
    room_type_hint = None
    if any(w in q for w in ["entire", "whole", "apartment", "house", "villa", "condo"]):
        room_type_hint = "Entire home/apt"
    elif any(w in q for w in ["private room", "private"]):
        room_type_hint = "Private room"
    elif any(w in q for w in ["shared", "hostel", "dorm"]):
        room_type_hint = "Shared room"

    # Amenity hints
    amenity_map = {
        "wifi": "Wifi",
        "wi-fi": "Wifi",
        "internet": "Wifi",
        "kitchen": "Kitchen",
        "parking": "Free parking on premises",
        "pool": "Pool",
        "gym": "Gym",
        "washer": "Washer",
        "dryer": "Dryer",
        "air conditioning": "Air conditioning",
        "ac": "Air conditioning",
        "workspace": "Workspace",
        "coffee": "Coffee maker",
        "dishwasher": "Dishwasher",
        "breakfast": "Breakfast",
        "elevator": "Elevator",
        "tv": "TV",
    }
    mentioned_amenities = [
        canonical for keyword, canonical in amenity_map.items()
        if keyword in q
    ]

    # Intent tags
    intent_tags = []
    for tag, keywords in {
        "quiet":    ["quiet", "peaceful", "calm", "tranquil"],
        "family":   ["family", "kids", "children", "child"],
        "luxury":   ["luxury", "upscale", "premium", "high-end"],
        "business": ["business", "workspace", "work", "professional", "remote work"],
        "romantic": ["romantic", "couple", "honeymoon", "date"],
        "budget":   ["budget", "cheap", "affordable", "inexpensive"],
    }.items():
        if any(kw in q for kw in keywords):
            intent_tags.append(tag)

    return {
        "mentioned_amenities": mentioned_amenities,
        "mentioned_city": mentioned_city,
        "budget_hint": budget_hint,
        "room_type_hint": room_type_hint,
        "intent_tags": intent_tags,
    }


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEER
# ─────────────────────────────────────────────────────────────────────────────

class FeatureEngineer:
    """
    Builds a feature matrix for (query, candidates) pairs.

    Parameters
    ----------
    listings : pd.DataFrame
        The full cleaned listings DataFrame (used for city-level statistics).
    """

    def __init__(self, listings: pd.DataFrame):
        self.listings = listings
        self._precompute_city_stats()

    def _precompute_city_stats(self) -> None:
        """Compute city-level statistics used for normalisation features."""
        self.city_stats = (
            self.listings.groupby("city")
            .agg(
                city_price_median=("price_usd", "median"),
                city_price_std=("price_usd", "std"),
                city_rating_median=("review_scores_rating", "median"),
            )
            .to_dict("index")
        )

    def _price_match_score(
        self, price: float, budget: Optional[float]
    ) -> float:
        """
        1.0 if price ≤ budget, decays linearly to 0 at 2×budget.
        Returns 0.5 if no budget specified (neutral).
        """
        if budget is None or budget <= 0:
            return 0.5
        if price <= budget:
            return 1.0
        overshoot = (price - budget) / budget   # 0 at budget, 1 at 2×budget
        return max(0.0, 1.0 - overshoot)

    def _amenity_match_ratio(
        self, amenities_list: list, mentioned_amenities: list
    ) -> float:
        """Fraction of query-mentioned amenities present in listing."""
        if not mentioned_amenities:
            return 0.5  # neutral when no amenities in query
        if not isinstance(amenities_list, list):
            return 0.0
        amenities_lower = [a.lower() for a in amenities_list]
        matches = sum(
            1 for a in mentioned_amenities
            if any(a.lower() in al for al in amenities_lower)
        )
        return matches / len(mentioned_amenities)

    def build_features(
        self,
        query: str,
        candidates: pd.DataFrame,
        semantic_scores: Optional[dict] = None,   # {listing_id: score}
        tfidf_scores: Optional[dict] = None,       # {listing_id: score}
        max_price: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Build a feature DataFrame with one row per candidate listing.

        Parameters
        ----------
        query       : Natural-language search query
        candidates  : DataFrame of candidate listings (output of hybrid_search)
        semantic_scores : mapping of listing_id → semantic similarity score
        tfidf_scores    : mapping of listing_id → TF-IDF similarity score
        max_price   : Maximum price filter from user (for price-match feature)

        Returns
        -------
        pd.DataFrame with columns = FEATURE_NAMES + ['listing_id', 'city']
        """
        intent = parse_query_intent(query)
        budget = max_price or intent["budget_hint"]
        mentioned_amenities = intent["mentioned_amenities"]
        room_type_hint = intent["room_type_hint"]
        city_hint = intent["mentioned_city"]

        feature_rows = []

        for _, row in candidates.iterrows():
            city = row.get("city", "")
            city_stat = self.city_stats.get(city, {})
            amenities_list = row.get("amenities_list", [])
            if not isinstance(amenities_list, list):
                # Restore from pipe-separated string if needed
                raw = row.get("amenities_list_str", "")
                amenities_list = raw.split("|") if raw else []

            price = _safe_float(row.get("price_usd"), 0.0)
            lid = row["id"]

            # ── Query-Listing similarity ──────────────────────────────────
            sem_score  = _safe_float((semantic_scores or {}).get(lid,
                         row.get("semantic_score_norm", row.get("semantic_score", 0.0))))
            kw_score   = _safe_float((tfidf_scores or {}).get(lid,
                         row.get("tfidf_score_norm", row.get("tfidf_score", 0.0))))
            hybrid_score = _safe_float(row.get("hybrid_score",
                           config.HYBRID_ALPHA * sem_score + (1 - config.HYBRID_ALPHA) * kw_score))

            amenity_match = self._amenity_match_ratio(amenities_list, mentioned_amenities)
            room_type_match = int(
                room_type_hint is not None
                and str(row.get("room_type", "")).lower() == room_type_hint.lower()
            )
            city_match = int(city_hint is not None and city == city_hint)

            # ── Price features ────────────────────────────────────────────
            price_match      = self._price_match_score(price, budget)
            price_pct        = _safe_float(row.get("price_percentile_city"), 0.5)
            price_vs_budget  = (price / budget) if (budget and budget > 0) else 0.5
            price_log        = _log_norm(price)
            over_budget      = int(budget is not None and price > budget)

            # ── Quality features ──────────────────────────────────────────
            rating           = _safe_float(row.get("review_scores_rating"), 0.0)
            rating_norm      = rating / 5.0
            n_reviews        = _safe_float(row.get("number_of_reviews"), 0.0)
            reviews_log      = _log_norm(n_reviews)
            clean_score      = _safe_float(row.get("review_scores_cleanliness"), 0.0) / 5.0
            loc_score        = _safe_float(row.get("review_scores_location"), 0.0) / 5.0
            value_score      = _safe_float(row.get("review_scores_value"), 0.0) / 5.0
            comm_score       = _safe_float(row.get("review_scores_communication"), 0.0) / 5.0

            # ── Host features ─────────────────────────────────────────────
            superhost        = int(bool(row.get("host_is_superhost", False)))
            resp_rate        = _safe_float(row.get("host_response_rate"), 0.5)
            acc_rate         = _safe_float(row.get("host_acceptance_rate"), 0.5)
            host_listings    = _log_norm(_safe_float(row.get("host_listings_count"), 1.0))
            instant_book     = int(bool(row.get("instant_bookable", False)))

            # ── Availability features ─────────────────────────────────────
            avail_365        = _safe_float(row.get("availability_365"), 180.0)
            avail_pct        = _safe_float(row.get("availability_pct"), 0.5)
            min_nights       = _safe_float(row.get("minimum_nights"), 1.0)
            min_nights_log   = _log_norm(min_nights)
            cal_avail_30     = _safe_float(row.get("cal_availability_rate_30"), 0.5)

            # ── Geographic features ───────────────────────────────────────
            lat = _safe_float(row.get("latitude"), 0.0)
            lon = _safe_float(row.get("longitude"), 0.0)
            city_centre = config.CITY_CENTRES.get(city, (lat, lon))
            try:
                dist_km = _haversine_km(lat, lon, city_centre[0], city_centre[1])
            except Exception:
                dist_km = 0.0
            dist_norm = 1.0 / (1.0 + dist_km)   # closer = higher score

            # ── Review-derived features ───────────────────────────────────
            avg_rev_len   = _log_norm(_safe_float(row.get("avg_review_length"), 0.0))
            revs_per_month = _safe_float(row.get("reviews_per_month"), 0.0)

            # ── Property features ─────────────────────────────────────────
            accommodates  = _safe_float(row.get("accommodates"), 2.0)
            bedrooms      = _safe_float(row.get("bedrooms"), 1.0)

            feature_rows.append({
                "listing_id":         lid,
                "city":               city,
                # Query-Listing
                "semantic_score":     sem_score,
                "tfidf_score":        kw_score,
                "hybrid_score":       hybrid_score,
                "amenity_match":      amenity_match,
                "room_type_match":    room_type_match,
                "city_match":         city_match,
                # Price
                "price_usd":          price,
                "price_log":          price_log,
                "price_match":        price_match,
                "price_percentile":   price_pct,
                "price_vs_budget":    price_vs_budget,
                "over_budget":        over_budget,
                # Quality
                "rating":             rating,
                "rating_norm":        rating_norm,
                "reviews_log":        reviews_log,
                "cleanliness_score":  clean_score,
                "location_score":     loc_score,
                "value_score":        value_score,
                "comm_score":         comm_score,
                # Host
                "superhost":          superhost,
                "response_rate":      resp_rate,
                "acceptance_rate":    acc_rate,
                "host_listings_log":  host_listings,
                "instant_bookable":   instant_book,
                # Availability
                "availability_365":   avail_365,
                "availability_pct":   avail_pct,
                "min_nights_log":     min_nights_log,
                "cal_avail_rate_30":  cal_avail_30,
                # Geographic
                "dist_city_centre_km": dist_km,
                "dist_norm":          dist_norm,
                # Review-derived
                "avg_review_length":  avg_rev_len,
                "reviews_per_month":  revs_per_month,
                # Property
                "accommodates":       accommodates,
                "bedrooms":           bedrooms,
            })

        return pd.DataFrame(feature_rows)


# ── Feature names (for XGBoost and SHAP) ─────────────────────────────────────
FEATURE_NAMES = [
    "semantic_score", "tfidf_score", "hybrid_score",
    "amenity_match", "room_type_match", "city_match",
    "price_usd", "price_log", "price_match", "price_percentile",
    "price_vs_budget", "over_budget",
    "rating", "rating_norm", "reviews_log",
    "cleanliness_score", "location_score", "value_score", "comm_score",
    "superhost", "response_rate", "acceptance_rate",
    "host_listings_log", "instant_bookable",
    "availability_365", "availability_pct",
    "min_nights_log", "cal_avail_rate_30",
    "dist_city_centre_km", "dist_norm",
    "avg_review_length", "reviews_per_month",
    "accommodates", "bedrooms",
]
