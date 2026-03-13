"""Systematic model comparison: run hierarchical + stability for multiple models."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.embed import load_or_compute_embeddings
from src.evaluate import compute_routing_accuracy
from src.hierarchical import cluster_per_topic
from src.routing import embed_label_prototypes, route_posts
from src.stability import run_stability_experiment


def _model_short_name(full_name: str) -> str:
    """Derive a filesystem-safe short name from a HuggingFace model path."""
    return full_name.rsplit("/", 1)[-1]


def run_model_comparison(
    cfg: dict[str, Any],
    df: pd.DataFrame,
    categories: np.ndarray | None,
    base_results_dir: Path,
) -> pd.DataFrame:
    """Run hierarchical pipeline + stability for each configured model.

    Args:
        cfg: Full config dict (must contain ``models`` list).
        df: Preprocessed DataFrame with ``text_clean`` and ``id`` columns.
        categories: Ground-truth category array (or None).
        base_results_dir: Root results directory.

    Returns:
        Summary DataFrame with one row per model.
    """
    models_cfg: list[dict[str, str]] = cfg.get("models", [])
    if not models_cfg:
        raise ValueError("config.yaml enthält keine 'models' Liste für den Vergleich")

    interest_labels: dict[str, list[str]] = cfg.get("interest_labels", {})
    if not interest_labels:
        raise ValueError("interest_labels in config.yaml fehlt oder ist leer")

    hier_cfg = cfg.get("hierarchical", {})
    batch_size = cfg.get("model", {}).get("batch_size", 32)

    comp_dir = base_results_dir / "model_comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []

    for model_def in models_cfg:
        model_name = model_def["name"]
        short = _model_short_name(model_name)
        model_dir = comp_dir / short
        model_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"MODELL: {model_name}")
        print(f"{'=' * 60}")

        t_start = time.perf_counter()

        # --- Embeddings (cached per model) ---
        emb_path = str(model_dir / "embeddings.npy")
        ids_path = str(model_dir / "embedding_ids.json")

        print("\n[1] Embeddings")
        embeddings, _ = load_or_compute_embeddings(
            df,
            text_col="text_clean",
            id_col="id",
            model_name=model_name,
            batch_size=batch_size,
            embeddings_path=emb_path,
            ids_path=ids_path,
        )
        print(f"  ↳ Shape: {embeddings.shape}")

        # --- Label prototypes (cached per model) ---
        print("\n[2] Label-Prototypen")
        proto_cache = str(model_dir / "proto_cache")
        prototypes = embed_label_prototypes(
            interest_labels,
            model_name=model_name,
            batch_size=batch_size,
            cache_dir=proto_cache,
        )

        # --- Hierarchical run ---
        print("\n[3] Hierarchical Pipeline")
        routing_df = route_posts(embeddings, prototypes, top_n=2)
        topic_labels = routing_df["topic_1"].values

        for topic in sorted(set(topic_labels)):
            n = int((topic_labels == topic).sum())
            print(f"    {topic}: {n} Posts")

        acc1: float | None = None
        acc2: float | None = None
        if categories is not None:
            acc_result = compute_routing_accuracy(routing_df, categories)
            acc1 = acc_result["accuracy_at_1"]
            acc2 = acc_result["accuracy_at_2"]
            print(f"  ↳ Accuracy@1={acc1}, Accuracy@2={acc2}")

        _hierarchy_df = cluster_per_topic(
            df, embeddings, topic_labels,
            method=hier_cfg.get("subcluster_method", "hdbscan"),
            method_params=hier_cfg.get("subcluster_params", {}),
            kmeans_fallback_k=hier_cfg.get("kmeans_fallback_k", 3),
            min_samples_for_hdbscan=hier_cfg.get("min_samples_for_hdbscan", 8),
            random_state=cfg.get("seed", 42),
        )

        # --- Stability ---
        print("\n[4] Stability Benchmark")
        stab_summary = run_stability_experiment(
            cfg, df, embeddings, prototypes, categories, model_dir,
        )

        t_elapsed = time.perf_counter() - t_start

        # --- Collect metrics ---
        s1 = stab_summary.get("stage1_routing", {})
        s2 = stab_summary.get("stage2_subcluster_ari", {})

        topic_aris = [
            v["mean_ari"] for v in s2.values()
            if v.get("mean_ari") is not None
        ]

        row: dict[str, Any] = {
            "model": model_name,
            "routing_acc1_mean": s1.get("accuracy_at_1_mean"),
            "routing_acc1_std": s1.get("accuracy_at_1_std"),
            "routing_acc2_mean": s1.get("accuracy_at_2_mean"),
            "single_run_acc1": acc1,
            "mean_subcluster_ari": round(float(np.mean(topic_aris)), 4) if topic_aris else None,
            "std_subcluster_ari": round(float(np.std(topic_aris)), 4) if topic_aris else None,
            "embedding_dim": embeddings.shape[1],
            "avg_runtime_seconds": round(t_elapsed, 1),
        }

        for topic in sorted(s2):
            row[f"ari_{topic}"] = s2[topic].get("mean_ari")

        summary_rows.append(row)

        model_metrics = {
            "model": model_name,
            "single_run": {"accuracy_at_1": acc1, "accuracy_at_2": acc2},
            "stability": stab_summary,
            "runtime_seconds": round(t_elapsed, 1),
        }
        (model_dir / "metrics.json").write_text(
            json.dumps(model_metrics, indent=2, default=str), encoding="utf-8",
        )

        print(f"\n  ✅ {short} fertig in {t_elapsed:.1f}s")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = comp_dir / "model_comparison_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n  ↳ Vergleichstabelle → {summary_path}")

    return summary_df
