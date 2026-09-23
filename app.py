"""
app.py
======
Streamlit UI for the Intelligent Airbnb Search & Ranking Engine.

Run with:
    streamlit run app.py
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
import config

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Airbnb Search Engine",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CACHED RESOURCE LOADERS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading search engine…")
def load_engine():
    from src.search import AirbnbSearchEngine
    engine = AirbnbSearchEngine()
    engine.load()
    return engine


@st.cache_resource(show_spinner="Loading ranking model…")
def load_ranker(engine):
    from src.features import FeatureEngineer
    from src.ranking import XGBRankingModel
    fe = FeatureEngineer(engine.listings)
    model = XGBRankingModel()
    try:
        model.load(fe)
        return model, fe, True
    except FileNotFoundError:
        return None, fe, False


@st.cache_data(show_spinner=False)
def load_eval_results():
    if config.EVAL_RESULTS_PATH.exists():
        with open(config.EVAL_RESULTS_PATH) as f:
            return json.load(f)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — LISTING CARD
# ─────────────────────────────────────────────────────────────────────────────

def render_listing_card(row: pd.Series, rank: int, show_xgb: bool = False):
    """Render a single listing result card with explanation panel."""

    with st.container():
        col_img, col_info, col_score = st.columns([1.8, 4, 1.5])

        # ── Image ──────────────────────────────────────────────────────────
        with col_img:
            pic_url = str(row.get("picture_url", "")).strip()
            if pic_url and pic_url.startswith("http"):
                st.image(pic_url, use_container_width=True)
            else:
                st.markdown("🏠", )

        # ── Info ───────────────────────────────────────────────────────────
        with col_info:
            name   = str(row.get("name", "Unnamed Listing"))[:80]
            city   = str(row.get("city", ""))
            nbhd   = str(row.get("neighbourhood_cleansed", ""))
            rtype  = str(row.get("room_type", ""))
            price  = row.get("price_usd", None)

            st.markdown(f"**#{rank} — {name}**")
            st.markdown(
                f"`{city}` &nbsp;·&nbsp; {nbhd} &nbsp;·&nbsp; {rtype}"
            )

            # Rating + reviews
            rating   = row.get("review_scores_rating", None)
            n_reviews = row.get("number_of_reviews", None)
            rating_str = (
                f"⭐ {rating:.2f} ({int(n_reviews)} reviews)"
                if pd.notna(rating) and pd.notna(n_reviews)
                else "No ratings yet"
            )
            st.markdown(rating_str)

            # Amenity badges
            amenities_raw = str(row.get("amenities_list_str", ""))
            amenities = [a.strip() for a in amenities_raw.split("|") if a.strip()][:8]
            if amenities:
                st.markdown(" &nbsp; ".join([f"`{a}`" for a in amenities]))

        # ── Score + Price ─────────────────────────────────────────────────
        with col_score:
            if price is not None and pd.notna(price):
                st.metric("Price / night", f"${price:.0f}")

            if show_xgb and "xgb_score" in row.index:
                st.metric("XGB Score", f"{float(row['xgb_score']):.3f}")
            elif "hybrid_score" in row.index:
                st.metric("Hybrid Score", f"{float(row['hybrid_score']):.3f}")

            superhost = row.get("host_is_superhost", False)
            if superhost:
                st.markdown("🏅 **Superhost**")

        # ── Explanation panel ─────────────────────────────────────────────
        with st.expander("💡 Why this listing?"):
            reasons = []

            # Semantic similarity
            sem = row.get("feat_semantic_score", row.get("semantic_score_norm", None))
            if sem is not None and pd.notna(sem):
                score_val = float(sem)
                label = "Strong" if score_val > 0.6 else "Good" if score_val > 0.35 else "Moderate"
                reasons.append(f"✅ {label} semantic match (score: {score_val:.2f})")

            # Budget match
            if price is not None and pd.notna(price):
                price_match = row.get("feat_price_match", None)
                if price_match is not None and float(price_match) >= 1.0:
                    reasons.append(f"✅ Within your budget (${price:.0f}/night)")
                elif price_match is not None and float(price_match) < 1.0:
                    reasons.append(f"⚠️ Slightly over budget (${price:.0f}/night)")

            # Rating
            if pd.notna(rating) and rating >= 4.5:
                reasons.append(f"✅ Highly rated ({rating:.2f} ⭐)")
            elif pd.notna(rating) and rating >= 4.0:
                reasons.append(f"✅ Well rated ({rating:.2f} ⭐)")

            # Amenity match
            amenity_match = row.get("feat_amenity_match", None)
            if amenity_match is not None and float(amenity_match) > 0.5:
                reasons.append(f"✅ Good amenity match ({float(amenity_match)*100:.0f}% of requested amenities)")

            # Superhost
            if superhost:
                reasons.append("✅ Superhost — experienced, highly-rated host")

            # Review count
            if pd.notna(n_reviews) and n_reviews >= 50:
                reasons.append(f"✅ Well-reviewed ({int(n_reviews)} reviews)")

            # Availability
            avail = row.get("availability_365", None)
            if avail is not None and pd.notna(avail) and float(avail) > 200:
                reasons.append(f"✅ Highly available ({int(avail)} days/year)")

            # Distance
            dist = row.get("feat_dist_city_centre_km", None)
            if dist is not None and pd.notna(dist) and float(dist) < 3.0:
                reasons.append(f"✅ Close to city centre ({float(dist):.1f} km)")

            if reasons:
                for r in reasons:
                    st.markdown(r)
            else:
                st.markdown("ℹ️ Matched by overall text similarity and quality signals.")

        st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Load resources ────────────────────────────────────────────────────
    engine = load_engine()
    ranker, fe, ranker_available = load_ranker(engine)

    # ── Tabs ──────────────────────────────────────────────────────────────
    tab_search, tab_eval, tab_about = st.tabs(["🔍 Search", "📊 Evaluation", "ℹ️ About"])

    # ══════════════════════════════════════════════════════════════════════
    # TAB 1 — SEARCH
    # ══════════════════════════════════════════════════════════════════════
    with tab_search:
        st.title("🏠 Intelligent Airbnb Search Engine")
        st.caption(
            "Multi-city search across Albany · Amsterdam · Antwerp · Asheville · Athens"
        )

        # ── Search bar ────────────────────────────────────────────────────
        query = st.text_input(
            label="What are you looking for?",
            placeholder="e.g. quiet apartment in Amsterdam with WiFi and kitchen under $150",
            key="search_query",
        )

        col_btn, col_mode = st.columns([1, 3])
        with col_btn:
            search_clicked = st.button("🔍 Search", type="primary", use_container_width=True)
        with col_mode:
            use_ranker = st.checkbox(
                "⚡ Use XGBoost re-ranking",
                value=ranker_available,
                disabled=not ranker_available,
                help="Requires running: python src/ranking.py --train",
            )

        # ── Sidebar filters ───────────────────────────────────────────────
        with st.sidebar:
            st.header("🔧 Filters")

            selected_cities = st.multiselect(
                "Cities",
                options=engine.get_city_options(),
                default=[],
                help="Leave blank to search all cities",
            )

            max_price = st.slider(
                "Max price per night ($)",
                min_value=10, max_value=1000, value=300, step=10,
            )

            min_rating = st.slider(
                "Minimum rating",
                min_value=1.0, max_value=5.0, value=3.5, step=0.5,
            )

            min_guests = st.number_input(
                "Minimum guests", min_value=1, max_value=16, value=1,
            )

            room_types = st.multiselect(
                "Room type",
                options=engine.get_room_type_options(),
                default=[],
            )

            nbhd_options = engine.get_neighbourhood_options(
                selected_cities if selected_cities else None
            )
            selected_nbhds = st.multiselect(
                "Neighbourhoods",
                options=nbhd_options[:100],  # cap for performance
                default=[],
            )

            selected_amenities = st.multiselect(
                "Must-have amenities",
                options=config.TOP_AMENITIES[:20],
                default=[],
            )

            st.divider()
            st.caption("📌 Hard filters (price, room type, guests) are applied after retrieval.")

        # ── Results ───────────────────────────────────────────────────────
        if search_clicked and query.strip():
            with st.spinner("Searching…"):
                t0 = time.perf_counter()

                city_filter = selected_cities if selected_cities else None

                candidates = engine.hybrid_search(
                    query=query,
                    top_k=config.TOP_K_RETRIEVE,
                    city_filter=city_filter,
                    max_price=max_price,
                    min_rating=min_rating,
                    room_types=room_types if room_types else None,
                    min_guests=min_guests if min_guests > 1 else None,
                    neighbourhoods=selected_nbhds if selected_nbhds else None,
                    amenity_filters=selected_amenities if selected_amenities else None,
                )

                if use_ranker and ranker is not None and not candidates.empty:
                    results = ranker.rank(query, candidates, fe, max_price=max_price)
                    results = results.head(config.TOP_K_RESULTS)
                    mode_label = "Hybrid + XGBoost"
                else:
                    results = candidates.head(config.TOP_K_RESULTS)
                    mode_label = "Hybrid (TF-IDF + Semantic)"

                elapsed_ms = (time.perf_counter() - t0) * 1000

            if results.empty:
                st.warning(
                    "No listings found matching your query and filters. "
                    "Try relaxing the filters or rephrasing your query."
                )
            else:
                st.success(
                    f"Found **{len(results)}** listings · Mode: `{mode_label}` · "
                    f"Latency: `{elapsed_ms:.0f} ms`"
                )

                for rank, (_, row) in enumerate(results.iterrows(), start=1):
                    render_listing_card(row, rank, show_xgb=use_ranker)

        elif search_clicked and not query.strip():
            st.error("Please enter a search query.")

        else:
            # Landing page examples
            st.info(
                "👆 Enter a natural-language query above. Examples:\n\n"
                "- *quiet apartment in Amsterdam with WiFi under $150*\n"
                "- *family-friendly place in Athens near the Acropolis*\n"
                "- *cheap private room in Asheville with parking*\n"
                "- *luxury entire home in Antwerp*"
            )

    # ══════════════════════════════════════════════════════════════════════
    # TAB 2 — EVALUATION
    # ══════════════════════════════════════════════════════════════════════
    with tab_eval:
        st.header("📊 Model Evaluation Results")
        st.caption(
            "**Note**: Labels are SYNTHETIC — not real Airbnb user behaviour data. "
            "Metrics measure relative improvement across model variants."
        )

        eval_results = load_eval_results()

        if eval_results:
            df_eval = pd.DataFrame(eval_results).set_index("model")
            st.dataframe(df_eval, use_container_width=True)

            # Bar chart — NDCG@10
            if "NDCG@10" in df_eval.columns:
                st.subheader("NDCG@10 by Model")
                st.bar_chart(df_eval["NDCG@10"])

            # Bar chart — MRR@10
            if "MRR@10" in df_eval.columns:
                st.subheader("MRR@10 by Model")
                st.bar_chart(df_eval["MRR@10"])

        else:
            st.warning(
                "No evaluation results found. Run:\n\n"
                "```bash\n"
                "python src/ranking.py --train\n"
                "python evaluation/evaluate.py\n"
                "```"
            )

    # ══════════════════════════════════════════════════════════════════════
    # TAB 3 — ABOUT
    # ══════════════════════════════════════════════════════════════════════
    with tab_about:
        st.header("ℹ️ About this Project")
        st.markdown("""
**Intelligent Airbnb Search & Ranking Engine** is a production-oriented portfolio project
demonstrating an end-to-end search and ranking pipeline for accommodation listings.

### Architecture

```
Natural Language Query
        │
   ┌────┴────┐
   ▼         ▼
TF-IDF    Sentence Transformer
           (all-MiniLM-L6-v2 + FAISS)
   │         │
   └────┬────┘
        ▼
  Hybrid Score (α=0.6·semantic + 0.4·TF-IDF)
        │
        ▼
  Hard Filters (price, city, room type)
        │
        ▼
  Feature Engineering (34 features)
        │
        ▼
  XGBoost Ranker (rank:ndcg)
        │
        ▼
  Top-10 Results + Explanations
```

### Dataset
- **Source**: [Inside Airbnb](https://insideairbnb.com/) — publicly available data
- **Cities**: Albany, Amsterdam, Antwerp, Asheville, Athens
- **Listings**: ~25,700 after cleaning
- **Files used**: listings, reviews, calendar, neighbourhoods

### Disclaimer
This project is **not affiliated with Airbnb Inc**. It uses independently
collected public data. Relevance labels used for training are **synthetic** —
not real user clicks or bookings.

### Stack
`Python` · `Sentence Transformers` · `FAISS` · `XGBoost` · `Streamlit` · `scikit-learn`
        """)

        # Dataset stats
        st.subheader("Dataset Statistics")
        city_counts = engine.listings["city"].value_counts().reset_index()
        city_counts.columns = ["City", "Listings"]
        st.dataframe(city_counts, use_container_width=False)


if __name__ == "__main__":
    main()
