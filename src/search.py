"""
src/search.py
=============
Search module for the Airbnb Ranking Engine.

Implements three retrieval strategies:
  1. TF-IDF keyword search  (lexical baseline)
  2. Sentence Transformer semantic search via FAISS
  3. Hybrid search  (α·semantic + (1-α)·tfidf)

Usage
-----
Build indexes (one-time):
    python src/search.py --build-index

Interactive search (for testing):
    python src/search.py --query "quiet apartment in Amsterdam with WiFi"
"""

import argparse
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")


# ─────────────────────────────────────────────────────────────────────────────
# LAZY IMPORTS — only pulled in when needed so app startup is fast
# ─────────────────────────────────────────────────────────────────────────────

def _import_faiss():
    try:
        import faiss
        return faiss
    except ImportError:
        raise ImportError("faiss-cpu is not installed. Run: pip install faiss-cpu")


def _import_sentence_transformers():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers is not installed. Run: pip install sentence-transformers"
        )


def _import_sklearn_tfidf():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel
    return TfidfVectorizer, linear_kernel


# ─────────────────────────────────────────────────────────────────────────────
# INDEX BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_index(listings: Optional[pd.DataFrame] = None) -> None:
    """
    Build and save:
      - TF-IDF vectorizer + matrix
      - Sentence Transformer embeddings (cached as .npy)
      - FAISS IndexFlatIP index
    """
    if listings is None:
        log.info("Loading listings_clean.parquet…")
        listings = pd.read_parquet(config.LISTINGS_CLEAN_PATH)

    config.ensure_dirs()

    texts = listings["listing_text"].fillna("").tolist()
    listing_ids = listings["id"].values
    log.info("Building index for %d listings…", len(listings))

    # ── TF-IDF ──────────────────────────────────────────────────────────────
    if config.TFIDF_VECTORIZER_PATH.exists() and config.TFIDF_MATRIX_PATH.exists():
        log.info("Found existing TF-IDF artifacts at %s — skipping re-fit.", config.TFIDF_VECTORIZER_PATH)
    else:
        log.info("Fitting TF-IDF vectorizer (max_features=%d)…", config.TFIDF_MAX_FEATURES)
        TfidfVectorizer, _ = _import_sklearn_tfidf()
        tfidf = TfidfVectorizer(
            max_features=config.TFIDF_MAX_FEATURES,
            ngram_range=config.TFIDF_NGRAM_RANGE,
            strip_accents="unicode",
            analyzer="word",
            sublinear_tf=True,
        )
        tfidf_matrix = tfidf.fit_transform(texts)

        with open(config.TFIDF_VECTORIZER_PATH, "wb") as f:
            pickle.dump(tfidf, f)
        sparse.save_npz(str(config.TFIDF_MATRIX_PATH), tfidf_matrix)
        log.info("TF-IDF: saved vectorizer and matrix (%s terms)", len(tfidf.vocabulary_))

    # ── Sentence Transformer embeddings ─────────────────────────────────────
    if config.EMBEDDINGS_MATRIX_PATH.exists() and config.FAISS_INDEX_PATH.exists() and config.LISTING_IDS_PATH.exists():
        log.info("Found existing FAISS index and embeddings at %s — skipping re-encode.", config.FAISS_INDEX_PATH)
        return

    log.info("Loading embedding model: %s…", config.EMBEDDING_MODEL_NAME)
    SentenceTransformer = _import_sentence_transformers()
    model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)

    import os
    import torch
    n_threads = max(1, os.cpu_count() or 4)
    torch.set_num_threads(n_threads)
    log.info("Configured PyTorch CPU threads: %d", n_threads)

    chk_path = config.EMBEDDINGS_DIR / "embeddings_checkpoint.npy"
    chunk_size = 2500
    all_embeddings = []

    start_idx = 0
    if chk_path.exists():
        try:
            cached = np.load(str(chk_path))
            if len(cached) > 0 and len(cached) < len(texts):
                start_idx = len(cached)
                all_embeddings.append(cached)
                log.info("Resuming embedding generation from checkpoint: %d/%d listings", start_idx, len(texts))
        except Exception as e:
            log.warning("Could not load checkpoint: %s. Starting fresh.", e)
            start_idx = 0

    log.info("Encoding %d listing texts (chunks of %d, batch_size=256)…", len(texts) - start_idx, chunk_size)
    for i in range(start_idx, len(texts), chunk_size):
        chunk_texts = texts[i : i + chunk_size]
        chunk_emb = model.encode(
            chunk_texts,
            batch_size=256,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        all_embeddings.append(chunk_emb)

        # Save checkpoint
        curr_stacked = np.vstack(all_embeddings)
        np.save(str(chk_path), curr_stacked)
        log.info("  Encoded %d / %d listings (%.1f%%)", min(i + chunk_size, len(texts)), len(texts), 100.0 * min(i + chunk_size, len(texts)) / len(texts))

    embeddings = np.vstack(all_embeddings)
    np.save(str(config.EMBEDDINGS_MATRIX_PATH), embeddings)
    np.save(str(config.LISTING_IDS_PATH), listing_ids)
    if chk_path.exists():
        try:
            chk_path.unlink()
        except Exception:
            pass
    log.info("Embeddings saved: shape %s", embeddings.shape)

    # ── FAISS index ──────────────────────────────────────────────────────────
    faiss = _import_faiss()
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # inner product on L2-normalised = cosine
    index.add(embeddings)
    faiss.write_index(index, str(config.FAISS_INDEX_PATH))
    log.info("FAISS index built with %d vectors (dim=%d)", index.ntotal, dim)
    log.info("Index building complete ✓")


# ─────────────────────────────────────────────────────────────────────────────
# SEARCH ENGINE CLASS
# ─────────────────────────────────────────────────────────────────────────────

class AirbnbSearchEngine:
    """
    Unified search engine supporting TF-IDF, semantic, and hybrid retrieval.

    Example
    -------
    engine = AirbnbSearchEngine()
    engine.load()
    results = engine.hybrid_search("quiet apartment in Amsterdam with WiFi", top_k=10)
    """

    def __init__(self):
        self.listings: Optional[pd.DataFrame] = None
        self.tfidf = None
        self.tfidf_matrix = None
        self.embeddings: Optional[np.ndarray] = None
        self.listing_ids: Optional[np.ndarray] = None
        self.faiss_index = None
        self.embed_model = None
        self._loaded = False

    # ── Loading ──────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load all pre-built indexes and the listings DataFrame into memory."""
        if self._loaded:
            return

        log.info("Loading search engine artifacts…")

        # Listings
        self.listings = pd.read_parquet(config.LISTINGS_CLEAN_PATH)
        # Restore amenities_list from pipe-separated string
        if "amenities_list_str" in self.listings.columns:
            self.listings["amenities_list"] = (
                self.listings["amenities_list_str"]
                .fillna("")
                .apply(lambda s: s.split("|") if s else [])
            )

        # TF-IDF
        with open(config.TFIDF_VECTORIZER_PATH, "rb") as f:
            self.tfidf = pickle.load(f)
        self.tfidf_matrix = sparse.load_npz(str(config.TFIDF_MATRIX_PATH))

        # Embeddings + FAISS
        self.embeddings = np.load(str(config.EMBEDDINGS_MATRIX_PATH))
        self.listing_ids = np.load(str(config.LISTING_IDS_PATH))
        faiss = _import_faiss()
        self.faiss_index = faiss.read_index(str(config.FAISS_INDEX_PATH))

        # Embedding model (for query encoding)
        SentenceTransformer = _import_sentence_transformers()
        self.embed_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)

        # Build id→row index for fast lookups
        self._id_to_row = {lid: i for i, lid in enumerate(self.listings["id"].values)}
        self._faiss_id_to_row = {lid: i for i, lid in enumerate(self.listing_ids)}

        self._loaded = True
        log.info("Search engine loaded: %d listings in index.", len(self.listings))

    # ── Query encoding ────────────────────────────────────────────────────────

    def _encode_query(self, query: str) -> np.ndarray:
        """Encode a text query into a normalised embedding vector."""
        vec = self.embed_model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        )
        return vec.astype(np.float32)

    # ── TF-IDF search ─────────────────────────────────────────────────────────

    def tfidf_search(
        self,
        query: str,
        top_k: int = config.TOP_K_RETRIEVE,
        city_filter: Optional[list] = None,
    ) -> pd.DataFrame:
        """Return top-k listings by TF-IDF cosine similarity."""
        from sklearn.metrics.pairwise import linear_kernel

        query_vec = self.tfidf.transform([query])
        scores = linear_kernel(query_vec, self.tfidf_matrix).flatten()

        top_indices = np.argsort(scores)[::-1][:top_k * 3]   # over-retrieve for filtering
        results = self.listings.iloc[top_indices].copy()
        results["tfidf_score"] = scores[top_indices]

        if city_filter:
            results = results[results["city"].isin(city_filter)]

        return results.head(top_k).reset_index(drop=True)

    # ── Semantic search ───────────────────────────────────────────────────────

    def semantic_search(
        self,
        query: str,
        top_k: int = config.TOP_K_RETRIEVE,
        city_filter: Optional[list] = None,
    ) -> pd.DataFrame:
        """Return top-k listings by semantic (FAISS) similarity."""
        query_vec = self._encode_query(query)

        # Over-retrieve if filtering
        retrieve_k = top_k * 3 if city_filter else top_k
        distances, indices = self.faiss_index.search(query_vec, retrieve_k)

        # Map FAISS result indices → listing IDs → DataFrame rows
        faiss_result_ids = self.listing_ids[indices[0]]
        scores = distances[0]

        rows = []
        for lid, score in zip(faiss_result_ids, scores):
            row_idx = self._id_to_row.get(lid)
            if row_idx is not None:
                rows.append((row_idx, score))

        if not rows:
            return pd.DataFrame()

        row_indices, sem_scores = zip(*rows)
        results = self.listings.iloc[list(row_indices)].copy()
        results["semantic_score"] = sem_scores

        if city_filter:
            results = results[results["city"].isin(city_filter)]

        return results.head(top_k).reset_index(drop=True)

    # ── Hybrid search ─────────────────────────────────────────────────────────

    def hybrid_search(
        self,
        query: str,
        top_k: int = config.TOP_K_RETRIEVE,
        alpha: float = config.HYBRID_ALPHA,
        city_filter: Optional[list] = None,
        max_price: Optional[float] = None,
        min_rating: Optional[float] = None,
        room_types: Optional[list] = None,
        min_guests: Optional[int] = None,
        neighbourhoods: Optional[list] = None,
        amenity_filters: Optional[list] = None,
    ) -> pd.DataFrame:
        """
        Hybrid retrieval:
          hybrid_score = α·semantic_score + (1-α)·tfidf_score

        Hard filters (price, room_type, city, guests) are applied after
        candidate retrieval so they don't silently drop semantically relevant
        listings from the candidate pool.

        Returns a DataFrame with columns including hybrid_score, semantic_score,
        tfidf_score, and all listing fields.
        """
        retrieve_k = config.TOP_K_RETRIEVE

        # ── Retrieve candidates from both systems ────────────────────────────
        sem_df = self.semantic_search(query, top_k=retrieve_k)
        kw_df  = self.tfidf_search(query, top_k=retrieve_k)

        # ── Normalise scores to [0, 1] ────────────────────────────────────────
        def _minmax(series: pd.Series) -> pd.Series:
            mn, mx = series.min(), series.max()
            if mx == mn:
                return pd.Series(np.ones(len(series)), index=series.index)
            return (series - mn) / (mx - mn)

        sem_df = sem_df.copy()
        kw_df  = kw_df.copy()

        if not sem_df.empty and "semantic_score" in sem_df.columns:
            sem_df["semantic_score_norm"] = _minmax(sem_df["semantic_score"])
        else:
            sem_df["semantic_score_norm"] = 0.0

        if not kw_df.empty and "tfidf_score" in kw_df.columns:
            kw_df["tfidf_score_norm"] = _minmax(kw_df["tfidf_score"])
        else:
            kw_df["tfidf_score_norm"] = 0.0

        # ── Merge on listing id ────────────────────────────────────────────────
        sem_subset = sem_df[["id", "semantic_score", "semantic_score_norm"]].copy()
        kw_subset  = kw_df[["id", "tfidf_score", "tfidf_score_norm"]].copy()

        merged = self.listings.copy()
        merged = merged.merge(sem_subset, on="id", how="left")
        merged = merged.merge(kw_subset,  on="id", how="left")

        # Only keep rows that appeared in at least one retrieval set
        candidate_ids = set(sem_df["id"].tolist()) | set(kw_df["id"].tolist())
        merged = merged[merged["id"].isin(candidate_ids)].copy()

        merged["semantic_score_norm"] = merged["semantic_score_norm"].fillna(0.0)
        merged["tfidf_score_norm"]    = merged["tfidf_score_norm"].fillna(0.0)

        merged["hybrid_score"] = (
            alpha * merged["semantic_score_norm"]
            + (1 - alpha) * merged["tfidf_score_norm"]
        )

        # ── Apply hard filters ─────────────────────────────────────────────────
        if city_filter:
            merged = merged[merged["city"].isin(city_filter)]
        if max_price is not None:
            merged = merged[merged["price_usd"] <= max_price]
        if min_rating is not None and "review_scores_rating" in merged.columns:
            merged = merged[
                merged["review_scores_rating"].isna()
                | (merged["review_scores_rating"] >= min_rating)
            ]
        if room_types:
            merged = merged[merged["room_type"].isin(room_types)]
        if min_guests is not None and "accommodates" in merged.columns:
            merged = merged[merged["accommodates"] >= min_guests]
        if neighbourhoods:
            merged = merged[merged["neighbourhood_cleansed"].isin(neighbourhoods)]
        if amenity_filters:
            for amenity in amenity_filters:
                col = "amenity_" + __import__("re").sub(
                    r"[^a-z0-9]", "_", amenity.lower()
                ).strip("_")
                if col in merged.columns:
                    merged = merged[merged[col] == 1]

        merged = merged.sort_values("hybrid_score", ascending=False)
        return merged.head(top_k).reset_index(drop=True)

    # ── Convenience ───────────────────────────────────────────────────────────

    def get_listing_by_id(self, listing_id) -> Optional[pd.Series]:
        """Return a single listing row by its ID."""
        row_idx = self._id_to_row.get(listing_id)
        if row_idx is None:
            return None
        return self.listings.iloc[row_idx]

    def get_city_options(self) -> list:
        return sorted(self.listings["city"].unique().tolist())

    def get_neighbourhood_options(self, cities: Optional[list] = None) -> list:
        df = self.listings
        if cities:
            df = df[df["city"].isin(cities)]
        return sorted(df["neighbourhood_cleansed"].dropna().unique().tolist())

    def get_room_type_options(self) -> list:
        return sorted(self.listings["room_type"].dropna().unique().tolist())


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _cli() -> None:
    parser = argparse.ArgumentParser(description="Airbnb Search Engine")
    parser.add_argument("--build-index", action="store_true",
                        help="Build TF-IDF + FAISS index from processed listings")
    parser.add_argument("--query", type=str, default=None,
                        help="Run a test search query")
    parser.add_argument("--mode", choices=["tfidf", "semantic", "hybrid"],
                        default="hybrid", help="Search mode")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    if args.build_index:
        build_index()

    if args.query:
        engine = AirbnbSearchEngine()
        engine.load()

        t0 = time.perf_counter()
        if args.mode == "tfidf":
            results = engine.tfidf_search(args.query, top_k=args.top_k)
        elif args.mode == "semantic":
            results = engine.semantic_search(args.query, top_k=args.top_k)
        else:
            results = engine.hybrid_search(args.query, top_k=args.top_k)
        elapsed = (time.perf_counter() - t0) * 1000

        print(f"\n🔍 Query: {args.query!r}  [{args.mode}]  ({elapsed:.0f} ms)\n")
        cols = ["id", "city", "name", "price_usd", "review_scores_rating",
                "room_type", "neighbourhood_cleansed"]
        score_col = {"tfidf": "tfidf_score", "semantic": "semantic_score",
                     "hybrid": "hybrid_score"}.get(args.mode, "hybrid_score")
        cols_present = [c for c in cols + [score_col] if c in results.columns]
        print(results[cols_present].to_string(index=False))


if __name__ == "__main__":
    _cli()
