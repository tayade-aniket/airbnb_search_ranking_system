"""
run_full_pipeline.py
====================
Master orchestrator script that executes the complete end-to-end ML & Search pipeline:
  Step 1: Build FAISS Dense Vector Index & TF-IDF (src/search.py)
  Step 2: Generate Queries, Extract Features & Train XGBRanker (src/ranking.py)
  Step 3: Run Information Retrieval & Ranking Evaluation (evaluation/evaluate.py)
  Step 4: Auto-update README.md with Real Measured Metrics

Usage:
    python run_full_pipeline.py
"""

import logging
import sys
import time
from pathlib import Path

# Insert root directory into sys.path
sys.path.insert(0, str(Path(__file__).parent))

import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("PipelineRunner")


def main():
    log.info("=" * 70)
    log.info("🚀 STARTING COMPLETE AIRBNB SEARCH & RANKING ENGINE PIPELINE")
    log.info("=" * 70)
    start_time = time.perf_counter()

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1: PREPROCESSING (if clean parquet does not exist)
    # ─────────────────────────────────────────────────────────────────────────
    if not config.LISTINGS_CLEAN_PATH.exists():
        log.info("\n>>> [STEP 1/4] Running Preprocessing...")
        from src.preprocessing import run_preprocessing
        run_preprocessing()
    else:
        log.info("\n>>> [STEP 1/4] Clean dataset already exists at %s", config.LISTINGS_CLEAN_PATH)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2: BUILD FAISS & TF-IDF INDEX
    # ─────────────────────────────────────────────────────────────────────────
    log.info("\n>>> [STEP 2/4] Building Search Index (TF-IDF + FAISS Dense Embeddings)...")
    from src.search import build_index
    build_index()

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3: TRAIN XGBOOST RANKER
    # ─────────────────────────────────────────────────────────────────────────
    log.info("\n>>> [STEP 3/4] Training XGBoost Learning-to-Rank Model...")
    from src.ranking import run_training
    run_training()

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 4: RUN EVALUATION BENCHMARK & UPDATE README
    # ─────────────────────────────────────────────────────────────────────────
    log.info("\n>>> [STEP 4/4] Running IR Evaluation Benchmark & Updating README.md...")
    from evaluation.evaluate import run_evaluation
    run_evaluation()

    total_time = time.perf_counter() - start_time
    log.info("=" * 70)
    log.info("🎉 COMPLETE PIPELINE FINISHED SUCCESSFULLY IN %.1f MINUTES (%.1f s)", total_time / 60.0, total_time)
    log.info("=" * 70)


if __name__ == "__main__":
    main()
