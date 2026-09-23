"""
api.py
======
FastAPI backend for the Intelligent Airbnb Search & Ranking Engine.

Endpoints
---------
  GET  /health                 — liveness check
  GET  /cities                 — list available cities
  POST /search                 — main search + ranking endpoint
  GET  /listing/{listing_id}   — single listing detail
  POST /recommend              — personalised ranking (basic)

Run with:
    uvicorn api:app --reload --host 0.0.0.0 --port 8000

Then call:
    curl -X POST http://localhost:8000/search \
         -H "Content-Type: application/json" \
         -d '{"query": "quiet apartment in Amsterdam", "max_price": 150}'
"""

import logging
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
import config

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

app = FastAPI(
    title="Airbnb Search & Ranking Engine",
    description="Intelligent multi-city accommodation search using hybrid retrieval and XGBoost ranking.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# STARTUP — load models once
# ─────────────────────────────────────────────────────────────────────────────

_engine = None
_ranker = None
_fe = None
_ranker_available = False


@app.on_event("startup")
async def startup_event():
    global _engine, _ranker, _fe, _ranker_available
    from src.search import AirbnbSearchEngine
    from src.features import FeatureEngineer
    from src.ranking import XGBRankingModel

    log.info("Loading search engine…")
    _engine = AirbnbSearchEngine()
    _engine.load()

    _fe = FeatureEngineer(_engine.listings)

    try:
        _ranker = XGBRankingModel()
        _ranker.load(_fe)
        _ranker_available = True
        log.info("XGBoost ranker loaded.")
    except FileNotFoundError:
        log.warning("XGBoost ranker not found — API will return hybrid results only.")
        _ranker_available = False


# ─────────────────────────────────────────────────────────────────────────────
# REQUEST / RESPONSE MODELS
# ─────────────────────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500,
                       example="quiet apartment in Amsterdam with WiFi")
    max_price: Optional[float] = Field(None, ge=0, le=10_000, example=150.0)
    min_rating: Optional[float] = Field(None, ge=0, le=5, example=4.0)
    min_guests: Optional[int] = Field(None, ge=1, le=20, example=2)
    cities: Optional[list[str]] = Field(None, example=["Amsterdam"])
    room_types: Optional[list[str]] = Field(None, example=["Entire home/apt"])
    neighbourhoods: Optional[list[str]] = Field(None)
    amenities: Optional[list[str]] = Field(None, example=["Wifi", "Kitchen"])
    use_ranker: bool = Field(True, description="Use XGBoost re-ranking if available")
    top_k: int = Field(config.TOP_K_RESULTS, ge=1, le=50)


class ListingResult(BaseModel):
    listing_id: int
    name: str
    city: str
    neighbourhood: Optional[str]
    room_type: Optional[str]
    price_usd: Optional[float]
    rating: Optional[float]
    num_reviews: Optional[int]
    superhost: Optional[bool]
    picture_url: Optional[str]
    latitude: Optional[float]
    longitude: Optional[float]
    hybrid_score: Optional[float]
    xgb_score: Optional[float]
    amenities: Optional[list[str]]
    listing_url: Optional[str]


class SearchResponse(BaseModel):
    query: str
    mode: str
    total_results: int
    latency_ms: float
    results: list[ListingResult]


class RecommendRequest(BaseModel):
    preferred_cities: Optional[list[str]] = None
    preferred_room_types: Optional[list[str]] = None
    max_price: Optional[float] = None
    preferred_amenities: Optional[list[str]] = None
    top_k: int = Field(10, ge=1, le=50)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _row_to_listing(row) -> ListingResult:
    amenities_raw = str(row.get("amenities_list_str", ""))
    amenities = [a.strip() for a in amenities_raw.split("|") if a.strip()][:15]

    return ListingResult(
        listing_id=int(row.get("id", 0)),
        name=str(row.get("name", ""))[:120],
        city=str(row.get("city", "")),
        neighbourhood=str(row.get("neighbourhood_cleansed", "")) or None,
        room_type=str(row.get("room_type", "")) or None,
        price_usd=float(row["price_usd"]) if row.get("price_usd") is not None else None,
        rating=float(row["review_scores_rating"]) if row.get("review_scores_rating") is not None else None,
        num_reviews=int(row["number_of_reviews"]) if row.get("number_of_reviews") is not None else None,
        superhost=bool(row.get("host_is_superhost", False)),
        picture_url=str(row.get("picture_url", "")) or None,
        latitude=float(row["latitude"]) if row.get("latitude") is not None else None,
        longitude=float(row["longitude"]) if row.get("longitude") is not None else None,
        hybrid_score=float(row["hybrid_score"]) if row.get("hybrid_score") is not None else None,
        xgb_score=float(row["xgb_score"]) if row.get("xgb_score") is not None else None,
        amenities=amenities or None,
        listing_url=str(row.get("listing_url", "")) or None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Liveness check."""
    return {
        "status": "ok",
        "listings_loaded": len(_engine.listings) if _engine else 0,
        "ranker_available": _ranker_available,
    }


@app.get("/cities")
def get_cities():
    """List available cities and listing counts."""
    if _engine is None:
        raise HTTPException(503, "Engine not loaded yet")
    counts = _engine.listings["city"].value_counts().to_dict()
    return {"cities": counts}


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    """
    Main search endpoint.

    Performs hybrid retrieval, applies hard filters, and optionally re-ranks
    with the XGBoost model.
    """
    if _engine is None:
        raise HTTPException(503, "Engine not loaded yet")

    t0 = time.perf_counter()

    candidates = _engine.hybrid_search(
        query=req.query,
        top_k=config.TOP_K_RETRIEVE,
        city_filter=req.cities or None,
        max_price=req.max_price,
        min_rating=req.min_rating,
        room_types=req.room_types or None,
        min_guests=req.min_guests,
        neighbourhoods=req.neighbourhoods or None,
        amenity_filters=req.amenities or None,
    )

    mode = "hybrid"
    if req.use_ranker and _ranker_available and _ranker is not None and not candidates.empty:
        results = _ranker.rank(req.query, candidates, _fe, max_price=req.max_price)
        mode = "hybrid+xgboost"
    else:
        results = candidates

    results = results.head(req.top_k)
    latency_ms = (time.perf_counter() - t0) * 1000

    listings = [_row_to_listing(row) for _, row in results.iterrows()]

    return SearchResponse(
        query=req.query,
        mode=mode,
        total_results=len(listings),
        latency_ms=round(latency_ms, 1),
        results=listings,
    )


@app.get("/listing/{listing_id}")
def get_listing(listing_id: int):
    """Return full detail for a single listing by ID."""
    if _engine is None:
        raise HTTPException(503, "Engine not loaded yet")

    row = _engine.get_listing_by_id(listing_id)
    if row is None:
        raise HTTPException(404, f"Listing {listing_id} not found")

    return _row_to_listing(row).model_dump()


@app.post("/recommend")
def recommend(req: RecommendRequest):
    """
    Basic personalised recommendations without a query.
    Returns high-rated, available listings matching user preferences.
    Note: This is a simple filter-based approach, not collaborative filtering.
    """
    if _engine is None:
        raise HTTPException(503, "Engine not loaded yet")

    df = _engine.listings.copy()

    if req.preferred_cities:
        df = df[df["city"].isin(req.preferred_cities)]
    if req.preferred_room_types:
        df = df[df["room_type"].isin(req.preferred_room_types)]
    if req.max_price:
        df = df[df["price_usd"] <= req.max_price]

    # Sort by rating × log(reviews+1) — simple quality score
    import numpy as np
    df["_quality"] = (
        df["review_scores_rating"].fillna(0)
        * np.log1p(df["number_of_reviews"].fillna(0))
    )
    df = df.sort_values("_quality", ascending=False).head(req.top_k)

    return {
        "mode": "filter-based (no query)",
        "results": [_row_to_listing(row).model_dump() for _, row in df.iterrows()],
    }
