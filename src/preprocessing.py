"""
src/preprocessing.py
====================
Data cleaning and feature-preparation pipeline for the Airbnb Ranking Engine.

Responsibilities
----------------
1. Load listings.csv from each of the 5 city folders and merge into one DataFrame.
2. Clean and normalise all relevant columns (price, ratings, host rates, text, amenities).
3. Build the `listing_text` field used by the embedding model.
4. Load and aggregate reviews.csv per listing (review count, avg length, recency).
5. Load and aggregate calendar.csv per listing (availability rate, avg calendar price).
6. Merge all three aggregates into a single enriched listings DataFrame.
7. Save processed outputs to data/processed/.

Usage
-----
    python src/preprocessing.py

All decisions that drop rows are logged with before/after counts so that no
data is silently discarded.
"""

import ast
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Allow running from repo root or from src/ ─────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  LISTINGS LOADING & MERGING
# ─────────────────────────────────────────────────────────────────────────────

def load_all_listings() -> pd.DataFrame:
    """Load listings.csv from each city folder and concatenate."""
    frames = []
    for city in config.CITIES:
        path = config.RAW_DIR / city / "listings.csv"
        if not path.exists():
            log.warning("listings.csv not found for %s — skipping.", city)
            continue
        df = pd.read_csv(path, low_memory=False)
        df["city"] = city
        log.info("Loaded %s: %d listings, %d columns", city, len(df), len(df.columns))
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True, sort=False)
    log.info("Combined dataset: %d listings, %d columns", len(combined), len(combined.columns))
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# 2.  COLUMN CLEANING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _parse_price(series: pd.Series) -> pd.Series:
    """Convert price strings like '$1,234.00' or '1234' to float."""
    return (
        series.astype(str)
        .str.replace(r"[\$,]", "", regex=True)
        .str.strip()
        .replace("nan", np.nan)
        .replace("", np.nan)
        .astype(float)
    )


def _parse_percentage(series: pd.Series) -> pd.Series:
    """Convert '95%' → 0.95, 'N/A' → NaN."""
    return (
        series.astype(str)
        .str.replace("%", "", regex=False)
        .str.strip()
        .replace({"N/A": np.nan, "nan": np.nan, "": np.nan})
        .astype(float)
        / 100.0
    )


def _parse_bool(series: pd.Series) -> pd.Series:
    """Convert 't'/'f' or 'True'/'False' to bool (NaN stays NaN)."""
    mapping = {"t": True, "f": False, "true": True, "false": False,
               "1": True, "0": False, "yes": True, "no": False}
    return series.astype(str).str.lower().map(mapping)


def _parse_amenities(series: pd.Series) -> pd.Series:
    """
    Parse amenities column which can be:
      - A JSON-style list string: '["Wifi", "Kitchen"]'
      - A plain comma-separated string: 'Wifi, Kitchen'
      - NaN / empty
    Returns a Series of Python lists.
    """
    def _parse_one(val):
        if pd.isna(val) or str(val).strip() == "":
            return []
        val = str(val).strip()
        # Try JSON / Python list literal
        if val.startswith("["):
            try:
                parsed = ast.literal_eval(val)
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed if x]
            except (ValueError, SyntaxError):
                pass
        # Fallback: comma-separated
        return [x.strip() for x in val.split(",") if x.strip()]

    return series.apply(_parse_one)


def _clean_text(series: pd.Series) -> pd.Series:
    """Strip HTML tags, collapse whitespace, fill NaN with empty string."""
    return (
        series.fillna("")
        .astype(str)
        .str.replace(r"<[^>]+>", " ", regex=True)   # remove HTML
        .str.replace(r"\s+", " ", regex=True)         # collapse whitespace
        .str.strip()
    )


def _fill_ratings_by_city(df: pd.DataFrame, rating_cols: list) -> pd.DataFrame:
    """Fill missing rating columns with city-level median (not global)."""
    for col in rating_cols:
        if col not in df.columns:
            continue
        city_medians = df.groupby("city")[col].transform("median")
        global_median = df[col].median()
        df[col] = df[col].fillna(city_medians).fillna(global_median)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3.  LISTING TEXT CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

def build_listing_text(row: pd.Series) -> str:
    """
    Construct a consolidated text representation of a listing for embedding.
    Gracefully handles any missing fields (e.g. Antwerp's limited schema).
    """
    parts = []

    # Name
    name = str(row.get("name", "")).strip()
    if name:
        parts.append(name)

    # Property type + city
    prop = str(row.get("property_type", "")).strip()
    city = str(row.get("city", "")).strip()
    nbhd = str(row.get("neighbourhood_cleansed", row.get("neighbourhood", ""))).strip()
    if prop or nbhd or city:
        location_str = " · ".join(x for x in [prop, nbhd, city] if x)
        parts.append(location_str)

    # Room type + capacity
    room = str(row.get("room_type", "")).strip()
    accommodates = row.get("accommodates", None)
    if room:
        cap_str = f"Sleeps {int(accommodates)}" if pd.notna(accommodates) else ""
        parts.append(" · ".join(x for x in [room, cap_str] if x))

    # Description (truncated to 500 chars to keep embedding manageable)
    desc = str(row.get("description", "")).strip()
    if desc:
        parts.append(desc[:500])

    # Neighbourhood overview (truncated to 200 chars)
    nbhd_desc = str(row.get("neighborhood_overview", "")).strip()
    if nbhd_desc:
        parts.append(nbhd_desc[:200])

    # Amenities
    amenities = row.get("amenities_list", [])
    if isinstance(amenities, list) and amenities:
        parts.append("Amenities: " + ", ".join(amenities[:25]))

    # Price signal
    price = row.get("price_usd", None)
    if pd.notna(price):
        parts.append(f"Price: ${price:.0f} per night")

    return ". ".join(parts).strip()


# ─────────────────────────────────────────────────────────────────────────────
# 4.  AMENITY BINARY FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def add_amenity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add a binary column for each amenity in config.TOP_AMENITIES."""
    for amenity in config.TOP_AMENITIES:
        col = "amenity_" + re.sub(r"[^a-z0-9]", "_", amenity.lower()).strip("_")
        df[col] = df["amenities_list"].apply(
            lambda lst: int(any(amenity.lower() in a.lower() for a in lst))
            if isinstance(lst, list) else 0
        )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5.  MAIN LISTINGS CLEANING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def clean_listings(df: pd.DataFrame) -> pd.DataFrame:
    """Full listings cleaning pipeline. Logs row counts at every filter step."""
    initial = len(df)
    log.info("Starting cleaning — %d rows", initial)

    # ── 5a. Deduplicate on listing id ──────────────────────────────────────
    df = df.drop_duplicates(subset=["id"])
    log.info("After dedup: %d rows (dropped %d)", len(df), initial - len(df))

    # ── 5b. Parse price ────────────────────────────────────────────────────
    if "price" in df.columns:
        df["price_usd"] = _parse_price(df["price"])
    else:
        df["price_usd"] = np.nan

    before = len(df)
    df = df[df["price_usd"].notna() & (df["price_usd"] >= config.PRICE_MIN) &
            (df["price_usd"] <= config.PRICE_MAX)]
    log.info("After price filter [%s–%s]: %d rows (dropped %d)",
             config.PRICE_MIN, config.PRICE_MAX, len(df), before - len(df))

    # ── 5c. Lat / lon ──────────────────────────────────────────────────────
    before = len(df)
    df = df[df["latitude"].notna() & df["longitude"].notna()]
    log.info("After lat/lon filter: %d rows (dropped %d)", len(df), before - len(df))

    # ── 5d. Parse host columns ─────────────────────────────────────────────
    if "host_response_rate" in df.columns:
        df["host_response_rate"] = _parse_percentage(df["host_response_rate"])
    if "host_acceptance_rate" in df.columns:
        df["host_acceptance_rate"] = _parse_percentage(df["host_acceptance_rate"])
    if "host_is_superhost" in df.columns:
        df["host_is_superhost"] = _parse_bool(df["host_is_superhost"])

    # ── 5e. Parse availability / review counts (ensure numeric) ───────────
    numeric_cols = [
        "accommodates", "bedrooms", "beds", "minimum_nights", "maximum_nights",
        "availability_30", "availability_60", "availability_90", "availability_365",
        "number_of_reviews", "number_of_reviews_ltm", "reviews_per_month",
        "calculated_host_listings_count", "host_listings_count",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── 5f. Parse rating columns ───────────────────────────────────────────
    rating_cols = [
        "review_scores_rating", "review_scores_accuracy",
        "review_scores_cleanliness", "review_scores_checkin",
        "review_scores_communication", "review_scores_location",
        "review_scores_value",
    ]
    for col in rating_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = _fill_ratings_by_city(df, rating_cols)

    # ── 5g. Clean text fields ──────────────────────────────────────────────
    text_cols = ["name", "description", "neighborhood_overview",
                 "host_about", "transit", "notes", "house_rules"]
    for col in text_cols:
        if col in df.columns:
            df[col] = _clean_text(df[col])

    # ── 5h. Parse amenities ────────────────────────────────────────────────
    if "amenities" in df.columns:
        df["amenities_list"] = _parse_amenities(df["amenities"])
    else:
        df["amenities_list"] = [[] for _ in range(len(df))]

    df = add_amenity_features(df)

    # ── 5i. Neighbourhood fallback ─────────────────────────────────────────
    if "neighbourhood_cleansed" not in df.columns:
        df["neighbourhood_cleansed"] = df.get("neighbourhood", pd.Series(dtype=str))
    df["neighbourhood_cleansed"] = df["neighbourhood_cleansed"].fillna("").astype(str)

    # ── 5j. Build listing_text ─────────────────────────────────────────────
    log.info("Building listing_text for all listings…")
    df["listing_text"] = df.apply(build_listing_text, axis=1)

    # ── 5k. Drop listings with essentially no text ─────────────────────────
    before = len(df)
    df = df[df["listing_text"].str.len() >= config.MIN_LISTING_TEXT_LEN]
    log.info("After text-length filter: %d rows (dropped %d)", len(df), before - len(df))

    # ── 5l. Availability percentage ────────────────────────────────────────
    if "availability_365" in df.columns:
        df["availability_pct"] = df["availability_365"] / 365.0

    # ── 5m. Price percentile within city ──────────────────────────────────
    df["price_percentile_city"] = df.groupby("city")["price_usd"].rank(pct=True)

    # ── 5n. Superhost fill ─────────────────────────────────────────────────
    if "host_is_superhost" in df.columns:
        df["host_is_superhost"] = df["host_is_superhost"].fillna(False).astype(bool)
    else:
        df["host_is_superhost"] = False

    # ── 5o. Instant bookable ───────────────────────────────────────────────
    if "instant_bookable" in df.columns:
        df["instant_bookable"] = _parse_bool(df["instant_bookable"]).fillna(False)
    else:
        df["instant_bookable"] = False

    log.info("Cleaning complete — %d listings retained (%.1f%% of original %d)",
             len(df), 100 * len(df) / initial, initial)
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  REVIEWS PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def process_reviews() -> pd.DataFrame:
    """
    Load reviews from all cities, compute per-listing aggregates, and return.
    Only keeps listing_id, date, comments columns to manage memory.
    Processes each city file individually before merging to avoid OOM.
    """
    log.info("Processing reviews…")
    agg_frames = []

    for city in config.CITIES:
        path = config.RAW_DIR / city / "reviews.csv"
        if not path.exists():
            log.warning("reviews.csv not found for %s — skipping.", city)
            continue

        # Read in chunks to handle large files (e.g. Athens = 294 MB)
        chunks = []
        chunk_iter = pd.read_csv(
            path,
            usecols=lambda c: c in {"listing_id", "date", "comments", "id"},
            chunksize=50_000,
            low_memory=False,
        )
        for chunk in chunk_iter:
            chunks.append(chunk)
        city_reviews = pd.concat(chunks, ignore_index=True)
        city_reviews["city"] = city

        log.info("  %s: %d reviews", city, len(city_reviews))

        # Per-listing aggregations
        city_reviews["comments"] = city_reviews["comments"].fillna("").astype(str)
        city_reviews["review_length"] = city_reviews["comments"].str.len()
        city_reviews["date"] = pd.to_datetime(city_reviews["date"], errors="coerce")

        agg = city_reviews.groupby("listing_id").agg(
            review_count_raw=("id", "count"),
            avg_review_length=("review_length", "mean"),
            last_review_date=("date", "max"),
            first_review_date=("date", "min"),
        ).reset_index()
        agg["city"] = city
        agg_frames.append(agg)

    if not agg_frames:
        log.warning("No review data found.")
        return pd.DataFrame(columns=["listing_id", "review_count_raw",
                                     "avg_review_length", "last_review_date", "city"])

    combined = pd.concat(agg_frames, ignore_index=True)
    log.info("Reviews aggregated: %d listings with review data", len(combined))
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# 7.  CALENDAR PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def process_calendar() -> pd.DataFrame:
    """
    Load calendar CSVs for all cities in chunks (files are large).
    Compute per-listing: availability_rate_30, availability_rate_90,
    avg_calendar_price.
    """
    log.info("Processing calendar files (this may take a few minutes)…")
    agg_frames = []

    for city in config.CITIES:
        path = config.RAW_DIR / city / "calendar.csv"
        if not path.exists():
            log.warning("calendar.csv not found for %s — skipping.", city)
            continue

        log.info("  Reading %s calendar…", city)
        city_agg = _aggregate_calendar_chunked(path, city)
        if city_agg is not None:
            agg_frames.append(city_agg)

    if not agg_frames:
        log.warning("No calendar data found.")
        return pd.DataFrame()

    combined = pd.concat(agg_frames, ignore_index=True)
    log.info("Calendar aggregated: %d listing-level records", len(combined))
    return combined


def _aggregate_calendar_chunked(path: Path, city: str) -> pd.DataFrame | None:
    """Read a calendar file in chunks and return per-listing aggregates."""
    # We only need listing_id, date, available, price
    needed_cols = {"listing_id", "date", "available", "price"}

    records = []
    try:
        chunk_iter = pd.read_csv(
            path,
            usecols=lambda c: c in needed_cols,
            chunksize=config.CALENDAR_CHUNKSIZE,
            low_memory=False,
        )
        for chunk in chunk_iter:
            chunk["price_num"] = _parse_price(chunk["price"]) if "price" in chunk.columns else np.nan
            chunk["available_bool"] = (
                chunk["available"].astype(str).str.lower().map({"t": 1, "f": 0})
                if "available" in chunk.columns else np.nan
            )
            chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")
            records.append(chunk[["listing_id", "date", "available_bool", "price_num"]])
    except Exception as e:
        log.error("Error reading calendar for %s: %s", city, e)
        return None

    if not records:
        return None

    df = pd.concat(records, ignore_index=True)

    # Availability rate over first 30 and 90 days from min date
    min_date = df["date"].min()
    df_30 = df[df["date"] <= min_date + pd.Timedelta(days=30)]
    df_90 = df[df["date"] <= min_date + pd.Timedelta(days=90)]

    avail_30 = df_30.groupby("listing_id")["available_bool"].mean().rename("cal_availability_rate_30")
    avail_90 = df_90.groupby("listing_id")["available_bool"].mean().rename("cal_availability_rate_90")
    avg_price = (
        df[df["available_bool"] == 0]  # unavailable = booked, price is meaningful
        .groupby("listing_id")["price_num"]
        .mean()
        .rename("avg_calendar_price")
    )

    agg = pd.concat([avail_30, avail_90, avg_price], axis=1).reset_index()
    agg["city"] = city
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# 8.  NEIGHBOURHOODS
# ─────────────────────────────────────────────────────────────────────────────

def load_neighbourhoods() -> pd.DataFrame:
    """Merge neighbourhoods.csv from all cities."""
    frames = []
    for city in config.CITIES:
        path = config.RAW_DIR / city / "neighbourhoods.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["city"] = city
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# 9.  MERGE & SAVE
# ─────────────────────────────────────────────────────────────────────────────

def merge_and_save(
    listings: pd.DataFrame,
    reviews_agg: pd.DataFrame,
    calendar_agg: pd.DataFrame,
) -> pd.DataFrame:
    """Merge review and calendar aggregates onto listings and save to parquet."""

    # Merge reviews
    if not reviews_agg.empty:
        reviews_agg = reviews_agg.rename(columns={"listing_id": "id"})
        reviews_agg["id"] = reviews_agg["id"].astype(listings["id"].dtype)
        listings = listings.merge(
            reviews_agg.drop(columns=["city"], errors="ignore"),
            on="id", how="left"
        )
        log.info("Merged review aggregates onto listings.")

    # Merge calendar
    if not calendar_agg.empty:
        calendar_agg = calendar_agg.rename(columns={"listing_id": "id"})
        calendar_agg["id"] = calendar_agg["id"].astype(listings["id"].dtype)
        listings = listings.merge(
            calendar_agg.drop(columns=["city"], errors="ignore"),
            on="id", how="left"
        )
        log.info("Merged calendar aggregates onto listings.")

    # ── Save ──────────────────────────────────────────────────────────────
    config.ensure_dirs()

    # Serialise list columns (amenities_list) as string for parquet compatibility
    listings["amenities_list_str"] = listings["amenities_list"].apply(
        lambda lst: "|".join(lst) if isinstance(lst, list) else ""
    )
    listings = listings.drop(columns=["amenities_list"], errors="ignore")

    # Drop the original raw 'price' string column (mixed types cause ArrowTypeError).
    # We already have price_usd (float) from cleaning.
    cols_to_drop = ["price", "amenities"]
    listings = listings.drop(columns=[c for c in cols_to_drop if c in listings.columns],
                             errors="ignore")

    # Coerce any remaining str/object columns that look numeric
    for col in listings.select_dtypes(include=["object", "str"]).columns:
        if col in ("listing_text", "amenities_list_str", "city", "name",
                   "description", "neighborhood_overview", "host_about",
                   "neighbourhood_cleansed", "neighbourhood", "room_type",
                   "property_type", "bathrooms_text", "host_name",
                   "host_response_time", "host_verifications", "listing_url",
                   "picture_url", "host_url", "host_picture_url",
                   "host_thumbnail_url", "host_location", "source",
                   "transit", "notes", "house_rules", "last_scraped",
                   "calendar_last_scraped", "first_review", "last_review",
                   "host_since", "calendar_updated", "last_review_date",
                   "first_review_date", "scrape_id", "license"):
            # Keep as string — convert NaN to empty string to avoid mixed types
            listings[col] = listings[col].fillna("").astype(str)
        else:
            listings[col] = pd.to_numeric(listings[col], errors="coerce")

    # Defragment the DataFrame (suppress PerformanceWarning from many inserts)
    listings = listings.copy()

    listings.to_parquet(config.LISTINGS_CLEAN_PATH, index=False)
    log.info("Saved listings_clean.parquet — shape: %s", listings.shape)

    return listings


# ─────────────────────────────────────────────────────────────────────────────
# 10. SUMMARY REPORT
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame) -> None:
    log.info("=" * 60)
    log.info("PREPROCESSING SUMMARY")
    log.info("=" * 60)
    log.info("Total listings: %d", len(df))
    city_counts = df["city"].value_counts()
    for city, count in city_counts.items():
        log.info("  %-12s : %d listings", city, count)
    log.info("Columns: %d", len(df.columns))
    log.info("Price range: $%.0f – $%.0f", df["price_usd"].min(), df["price_usd"].max())
    log.info("listing_text — avg length: %.0f chars", df["listing_text"].str.len().mean())
    missing_text = (df["listing_text"].str.len() < 20).sum()
    log.info("Listings with very short text (<20 chars): %d", missing_text)
    log.info("Antwerp schema note: amenities_list will be empty for Antwerp listings")
    log.info("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run_preprocessing() -> pd.DataFrame:
    """Full preprocessing pipeline. Returns the final cleaned DataFrame."""
    log.info("╔══════════════════════════════════════════════════╗")
    log.info("║       Airbnb Ranking Engine — Preprocessing      ║")
    log.info("╚══════════════════════════════════════════════════╝")

    # Ensure all output directories exist before any saves
    config.ensure_dirs()

    # Step 1 — Load & clean listings
    raw = load_all_listings()
    clean = clean_listings(raw)

    # Step 2 — Reviews
    reviews_agg = process_reviews()
    if not reviews_agg.empty:
        reviews_agg.to_parquet(config.REVIEWS_CLEAN_PATH, index=False)
        log.info("Saved reviews_clean.parquet")

    # Step 3 — Calendar
    calendar_agg = process_calendar()
    if not calendar_agg.empty:
        calendar_agg.to_parquet(config.CALENDAR_FEATURES_PATH, index=False)
        log.info("Saved calendar_features.parquet")

    # Step 4 — Neighbourhoods
    nbhd = load_neighbourhoods()
    if not nbhd.empty:
        nbhd.to_csv(config.NEIGHBOURHOODS_PATH, index=False)
        log.info("Saved neighbourhoods_combined.csv")

    # Step 5 — Merge & save
    final = merge_and_save(clean, reviews_agg, calendar_agg)
    print_summary(final)

    return final


if __name__ == "__main__":
    run_preprocessing()
