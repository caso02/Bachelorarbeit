#!/usr/bin/env python3
"""End-to-end pipeline: preprocess → embed → similarity → cluster → evaluate → report."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import yaml

warnings.filterwarnings("ignore", message=".*divide by zero.*matmul.*")
warnings.filterwarnings("ignore", message=".*overflow.*matmul.*")
warnings.filterwarnings("ignore", message=".*invalid value.*matmul.*")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess import load_posts, preprocess
from src.embed import load_or_compute_embeddings
from src.similarity import find_top_k
from src.cluster import run_clustering, run_dbscan_grid_search
from src.evaluate import (
    compute_metrics,
    compute_weak_label_metrics,
    extract_cluster_keywords,
    get_representative_examples,
)
from src.report import save_metrics, save_cluster_assignments, save_examples_markdown


def _resolve_path(path: str) -> Path:
    """Resolve a path relative to the project root."""
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


def main(config_path: str) -> None:
    """Run the full pipeline based on a YAML config."""
    cfg_file = Path(config_path)
    if not cfg_file.exists():
        raise FileNotFoundError(f"Config nicht gefunden: {cfg_file}")

    with open(cfg_file, encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f)

    seed = cfg.get("seed", 42)
    np.random.seed(seed)

    # --- 1. Load & Preprocess ---
    print("\n[1/6] Daten laden & bereinigen")
    data_cfg = cfg["data"]
    pre_cfg = cfg["preprocessing"]

    df = load_posts(
        path=_resolve_path(data_cfg["input_path"]),
        text_col=data_cfg["text_column"],
        id_col=data_cfg["id_column"],
    )
    print(f"  ↳ {len(df)} Posts geladen")

    df = preprocess(
        df,
        text_col=data_cfg["text_column"],
        remove_urls=pre_cfg.get("remove_urls", True),
        remove_mentions=pre_cfg.get("remove_mentions", True),
        lowercase=pre_cfg.get("lowercase", False),
        min_text_length=pre_cfg.get("min_text_length", 5),
        language_filter=data_cfg.get("language_filter"),
    )
    print(f"  ↳ {len(df)} Posts nach Preprocessing")

    has_categories = data_cfg.get("category_column") and data_cfg["category_column"] in df.columns
    if has_categories:
        categories = df[data_cfg["category_column"]].values
        print(f"  ↳ Weak labels vorhanden: {len(set(categories))} Kategorien")

    # --- 2. Embeddings ---
    print("\n[2/6] Embeddings erzeugen")
    model_cfg = cfg["model"]
    out_cfg = cfg["output"]

    embeddings, _ids = load_or_compute_embeddings(
        df,
        text_col="text_clean",
        id_col=data_cfg["id_column"],
        model_name=model_cfg["name"],
        batch_size=model_cfg.get("batch_size", 32),
        embeddings_path=str(_resolve_path(out_cfg["embeddings_path"])),
        ids_path=str(_resolve_path(out_cfg["ids_path"])),
    )
    print(f"  ↳ Embedding-Shape: {embeddings.shape}")

    # --- 3. Similarity Search ---
    print("\n[3/6] Similarity Search")
    sim_cfg = cfg["similarity"]
    query_idx = sim_cfg.get("query_index", 0)
    top_k = sim_cfg.get("top_k", 5)

    query_text = df.iloc[query_idx]["text_clean"]
    print(f"  ↳ Query (Index {query_idx}): \"{query_text}\"")

    sim_results = find_top_k(
        embeddings, query_idx, top_k=top_k, df=df,
        text_col="text_clean", id_col=data_cfg["id_column"],
    )
    print(f"  ↳ Top-{top_k} ähnliche Posts:")
    for _, row in sim_results.iterrows():
        print(f"     #{row['rank']}  (sim={row['similarity']:.4f})  {row['text_clean'][:80]}")

    # --- 4. Clustering ---
    print("\n[4/6] Clustering")
    cluster_cfg = cfg["clustering"]
    all_metrics: dict[str, Any] = {}
    cluster_labels: dict[str, np.ndarray] = {}
    cluster_details: dict[str, dict[str, Any]] = {}

    eval_cfg = cfg["evaluation"]
    top_kw = eval_cfg.get("top_keywords", 5)
    n_examples = eval_cfg.get("representative_examples", 3)

    for method_def in cluster_cfg["methods"]:
        name = method_def["name"]
        params = method_def.get("params", {})

        if name == "dbscan" and "grid_search" in method_def:
            gs = method_def["grid_search"]
            eps_values = gs["eps_values"]
            ms_values = gs["min_samples_values"]
            metric = params.get("metric", "cosine")

            print(f"\n  → DBSCAN GRID SEARCH ({len(eps_values)}×{len(ms_values)} Kombinationen, metric={metric})")
            grid_df, best = run_dbscan_grid_search(
                embeddings, eps_values=eps_values,
                min_samples_values=ms_values, metric=metric,
            )

            grid_path = _resolve_path(out_cfg.get("dbscan_grid_path", "results/dbscan_grid.csv"))
            grid_path.parent.mkdir(parents=True, exist_ok=True)
            grid_df.to_csv(grid_path, index=False)
            print(f"    Grid-Ergebnisse → {grid_path}")
            print(f"    Beste Kombination: eps={best['eps']}, min_samples={best['min_samples']}")

            labels = best["labels"]
            metrics = best["metrics"]
            metrics["best_eps"] = best["eps"]
            metrics["best_min_samples"] = best["min_samples"]
        else:
            print(f"\n  → {name.upper()} (params={params})")
            labels = run_clustering(embeddings, method=name, params=params)
            metrics = compute_metrics(embeddings, labels)

        unique = set(labels)
        unique.discard(-1)
        print(f"    Cluster gefunden: {len(unique)} (Noise: {(labels == -1).sum()})")
        print(f"    Silhouette={metrics.get('silhouette')}, Davies-Bouldin={metrics.get('davies_bouldin')}")

        cluster_labels[name] = labels
        all_metrics[name] = metrics

        keywords = extract_cluster_keywords(df, labels, text_col="text_clean", top_n=top_kw)
        examples = get_representative_examples(
            df, embeddings, labels,
            text_col="text_clean", id_col=data_cfg["id_column"],
            n_examples=n_examples,
        )

        method_details: dict[str, Any] = {}
        for cid in sorted(set(labels)):
            method_details[str(cid)] = {
                "keywords": keywords.get(cid, []),
                "examples": examples.get(cid, []),
            }
        cluster_details[name] = method_details

    # --- 5. Weak-label evaluation ---
    if has_categories:
        print("\n[5/6] Weak-Label Evaluation (NMI, ARI, Purity)")
        for name, labels in cluster_labels.items():
            wl = compute_weak_label_metrics(labels, categories)
            all_metrics[name]["weak_label"] = wl
            print(f"  → {name}: NMI={wl['nmi']}, ARI={wl['ari']}, "
                  f"Purity={wl['purity']}, Purity(+noise)={wl['purity_including_noise']}")
    else:
        print("\n[5/6] Weak-Label Evaluation – übersprungen (keine category-Spalte)")

    # --- 6. Reports ---
    print("\n[6/6] Reports generieren")

    save_metrics(all_metrics, _resolve_path(out_cfg["metrics_path"]))
    save_cluster_assignments(df, cluster_labels, _resolve_path(out_cfg["clusters_path"]))
    save_examples_markdown(cluster_details, _resolve_path(out_cfg["examples_path"]))

    print("\n✅ Pipeline abgeschlossen!")
    print(f"   → {out_cfg['metrics_path']}")
    print(f"   → {out_cfg['clusters_path']}")
    print(f"   → {out_cfg['examples_path']}")
    if any("grid_search" in m for m in cluster_cfg["methods"] if m["name"] == "dbscan"):
        print(f"   → {out_cfg.get('dbscan_grid_path', 'results/dbscan_grid.csv')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SBERT Social-Media Pipeline")
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/config.yaml",
        help="Pfad zur YAML-Konfigurationsdatei",
    )
    args = parser.parse_args()
    main(args.config)
