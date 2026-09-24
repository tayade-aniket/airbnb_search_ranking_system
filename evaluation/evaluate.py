"""
evaluation/evaluate.py
======================
Evaluation pipeline for all 4 retrieval/ranking models.

Metrics computed
----------------
  Retrieval:   Recall@K (K=10, 50, 100), MRR@10
  Ranking:     NDCG@5, NDCG@10
  System:      Avg latency, P95 latency (ms)

Models compared
---------------
  1. TF-IDF keyword search
  2. Semantic search (Sentence Transformer + FAISS)
  3. Hybrid search (α·semantic + (1-α)·tfidf)
  4. Hybrid + XGBoost ranker

Usage
-----
    python evaluation/evaluate.py

Outputs
-------
    results/evaluation_results.json  — machine-readable metrics
    Printed table to stdout
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")


# ─────────────────────────────────────────────────────────────────────────────
# METRIC IMPLEMENTATIONS
# ─────────────────────────────────────────────────────────────────────────────

def recall_at_k(retrieved_ids: list, relevant_ids: set, k: int) -> float:
    """Recall@K: fraction of relevant items found in top-K results."""
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    return len(top_k & relevant_ids) / len(relevant_ids)


def reciprocal_rank(retrieved_ids: list, relevant_ids: set) -> float:
    """Reciprocal rank of the first relevant item in the list."""
    for rank, item_id in enumerate(retrieved_ids, start=1):
        if item_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def dcg_at_k(relevances: list, k: int) -> float:
    """Discounted Cumulative Gain @ K."""
    relevances = relevances[:k]
    return sum(
        rel / np.log2(rank + 2)
        for rank, rel in enumerate(relevances)
    )


def ndcg_at_k(retrieved_ids: list, relevance_map: dict, k: int) -> float:
    """
    NDCG@K using a graded relevance map {listing_id: grade}.
    Ideal ranking is the sorted order of all available grades.
    """
    if not relevance_map:
        return 0.0

    actual_rels = [relevance_map.get(lid, 0) for lid in retrieved_ids[:k]]
    ideal_rels  = sorted(relevance_map.values(), reverse=True)[:k]

    dcg  = dcg_at_k(actual_rels, k)
    idcg = dcg_at_k(ideal_rels, k)

    return dcg / idcg if idcg > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATOR
# ─────────────────────────────────────────────────────────────────────────────

class SearchEvaluator:
    """
    Runs all 4 model variants on the held-out test query set and computes metrics.

    Uses the same test queries / relevance labels produced during training
    (stored in models/ranker/test_features.parquet).
    """

    def __init__(self):
        self.engine = None
        self.ranker = None
        self.fe = None
        self.test_df: pd.DataFrame | None = None

    def load(self) -> None:
        from src.search import AirbnbSearchEngine
        from src.features import FeatureEngineer
        from src.ranking import XGBRankingModel

        log.info("Loading search engine…")
        self.engine = AirbnbSearchEngine()
        self.engine.load()

        self.fe = FeatureEngineer(self.engine.listings)

        log.info("Loading XGBoost ranker…")
        self.ranker = XGBRankingModel()
        self.ranker.load(self.fe)

        log.info("Loading test features from %s…", config.TEST_FEATURES_PATH)
        self.test_df = pd.read_parquet(config.TEST_FEATURES_PATH)
        log.info("Test set: %d (query, listing) pairs across %d queries",
                 len(self.test_df), self.test_df["query_id"].nunique())

    def _get_relevant_ids(self, qid: str, min_relevance: int = 2) -> set:
        """Return set of listing_ids with relevance >= min_relevance for a query."""
        rows = self.test_df[self.test_df["query_id"] == qid]
        return set(rows[rows["relevance"] >= min_relevance]["listing_id"].tolist())

    def _get_relevance_map(self, qid: str) -> dict:
        """Return {listing_id: relevance_grade} for a query."""
        rows = self.test_df[self.test_df["query_id"] == qid]
        return dict(zip(rows["listing_id"], rows["relevance"]))

    def _evaluate_model(
        self,
        model_name: str,
        retrieval_fn,
        k_values: list = [10, 50, 100],
        ndcg_k_values: list = [5, 10],
        n_latency_samples: int = 50,
    ) -> dict:
        """
        Evaluate a single retrieval/ranking model on all test queries.

        Parameters
        ----------
        model_name    : Human-readable model name
        retrieval_fn  : callable(query_str) → list of listing_ids in ranked order
        """
        log.info("Evaluating: %s", model_name)

        test_qids = self.test_df["query_id"].unique()
        # Load query text from saved queries file
        with open(config.QUERIES_PATH) as f:
            all_queries = {q["query_id"]: q for q in json.load(f)}

        recall_scores  = {k: [] for k in k_values}
        mrr_scores     = []
        ndcg_scores    = {k: [] for k in ndcg_k_values}
        latencies_ms   = []

        for qid in test_qids:
            q_info = all_queries.get(qid)
            if q_info is None:
                continue
            query    = q_info["query"]
            relevant = self._get_relevant_ids(qid)
            rel_map  = self._get_relevance_map(qid)

            if not relevant:
                continue

            t0 = time.perf_counter()
            try:
                retrieved_ids = retrieval_fn(q_info)
            except Exception as e:
                log.warning("Retrieval failed for query %s: %s", qid, e)
                continue
            latency_ms = (time.perf_counter() - t0) * 1000
            latencies_ms.append(latency_ms)

            for k in k_values:
                recall_scores[k].append(recall_at_k(retrieved_ids, relevant, k))
            mrr_scores.append(reciprocal_rank(retrieved_ids, relevant))
            for k in ndcg_k_values:
                ndcg_scores[k].append(ndcg_at_k(retrieved_ids, rel_map, k))

        results = {"model": model_name}
        for k in k_values:
            key = f"Recall@{k}"
            results[key] = round(np.mean(recall_scores[k]), 4) if recall_scores[k] else 0.0
        results["MRR@10"] = round(np.mean(mrr_scores), 4) if mrr_scores else 0.0
        for k in ndcg_k_values:
            results[f"NDCG@{k}"] = round(np.mean(ndcg_scores[k]), 4) if ndcg_scores[k] else 0.0

        if latencies_ms:
            results["Avg_Latency_ms"]  = round(float(np.mean(latencies_ms)), 1)
            results["P95_Latency_ms"]  = round(float(np.percentile(latencies_ms, 95)), 1)
        else:
            results["Avg_Latency_ms"]  = None
            results["P95_Latency_ms"]  = None

        log.info("  %s → Recall@100=%.3f  MRR@10=%.3f  NDCG@10=%.3f  P95=%.0f ms",
                 model_name,
                 results.get("Recall@100", 0),
                 results.get("MRR@10", 0),
                 results.get("NDCG@10", 0),
                 results.get("P95_Latency_ms") or 0)
        return results

    # ── Per-model retrieval functions ─────────────────────────────────────────

    def _tfidf_retrieve(self, q_info: dict) -> list:
        res = self.engine.tfidf_search(
            q_info["query"],
            top_k=100,
            city_filter=[q_info["city"]] if q_info.get("city") else None,
        )
        return res["id"].tolist()

    def _semantic_retrieve(self, q_info: dict) -> list:
        res = self.engine.semantic_search(
            q_info["query"],
            top_k=100,
            city_filter=[q_info["city"]] if q_info.get("city") else None,
        )
        return res["id"].tolist()

    def _hybrid_retrieve(self, q_info: dict) -> list:
        res = self.engine.hybrid_search(
            q_info["query"],
            top_k=100,
            city_filter=[q_info["city"]] if q_info.get("city") else None,
            max_price=q_info.get("max_price"),
        )
        return res["id"].tolist()

    def _hybrid_xgb_retrieve(self, q_info: dict) -> list:
        candidates = self.engine.hybrid_search(
            q_info["query"],
            top_k=config.TOP_K_RETRIEVE,
            city_filter=[q_info["city"]] if q_info.get("city") else None,
            max_price=q_info.get("max_price"),
        )
        ranked = self.ranker.rank(
            q_info["query"], candidates, self.fe,
            max_price=q_info.get("max_price"),
        )
        return ranked["id"].tolist()

    # ── Run all ───────────────────────────────────────────────────────────────

    def run_all(self) -> list[dict]:
        models = [
            ("TF-IDF",           self._tfidf_retrieve),
            ("Semantic Search",  self._semantic_retrieve),
            ("Hybrid Search",    self._hybrid_retrieve),
            ("Hybrid + XGBoost", self._hybrid_xgb_retrieve),
        ]

        all_results = []
        for name, fn in models:
            result = self._evaluate_model(name, fn)
            all_results.append(result)

        return all_results


# ─────────────────────────────────────────────────────────────────────────────
# SAVE & PRINT
# ─────────────────────────────────────────────────────────────────────────────

def print_results_table(results: list[dict]) -> None:
    """Print a formatted comparison table to stdout."""
    df = pd.DataFrame(results)
    df = df.set_index("model")

    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(df.to_string())
    print("=" * 80)
    print("\nNote: Labels are SYNTHETIC — not real Airbnb user behaviour data.")
    print("These metrics measure improvement relative to baselines, not absolute quality.")


def update_readme_with_metrics(results: list[dict]) -> None:
    """Auto-update the Evaluation benchmark table in README.md with real measured metrics."""
    readme_path = config.ROOT_DIR / "README.md"
    if not readme_path.exists():
        return

    table_lines = [
        "| Approach | Recall@100 | MRR@10 | NDCG@10 | Avg Latency | P95 Latency |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        model = r.get("model", "")
        rec100 = f"{r.get('Recall@100', 0):.3f}"
        mrr10 = f"{r.get('MRR@10', 0):.3f}"
        ndcg10 = f"{r.get('NDCG@10', 0):.3f}"
        avg_lat = f"{r.get('Avg_Latency_ms', 0):.1f} ms" if r.get('Avg_Latency_ms') is not None else "—"
        p95_lat = f"{r.get('P95_Latency_ms', 0):.1f} ms" if r.get('P95_Latency_ms') is not None else "—"
        table_lines.append(f"| {model} | {rec100} | {mrr10} | {ndcg10} | {avg_lat} | {p95_lat} |")

    new_table_str = "\n".join(table_lines)

    content = readme_path.read_text(encoding="utf-8")
    import re
    pattern = r"(## Evaluation\s*\n\n)(?:>.*?\n\n)?\| Approach \|[\s\S]*?(?=\n\n---|\Z)"
    replacement = r"\1" + new_table_str

    if re.search(pattern, content):
        updated_content = re.sub(pattern, replacement, content)
        readme_path.write_text(updated_content, encoding="utf-8")
        log.info("Auto-updated README.md evaluation table with real metrics.")
    else:
        log.warning("Could not find evaluation table pattern in README.md to replace.")


def save_results(results: list[dict]) -> None:
    config.ensure_dirs()
    with open(config.EVAL_RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Evaluation results saved to %s", config.EVAL_RESULTS_PATH)
    update_readme_with_metrics(results)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation() -> None:
    log.info("╔══════════════════════════════════════════════════╗")
    log.info("║     Airbnb Ranking Engine — Evaluation           ║")
    log.info("╚══════════════════════════════════════════════════╝")

    evaluator = SearchEvaluator()
    evaluator.load()

    results = evaluator.run_all()
    print_results_table(results)
    save_results(results)


if __name__ == "__main__":
    run_evaluation()

