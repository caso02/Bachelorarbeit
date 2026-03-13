#!/usr/bin/env python3
"""End-to-end pipeline: flat or hierarchical (two-stage) mode."""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# Load .env if present (sets GOOGLE_API_KEY etc. without requiring python-dotenv)
_env_file = Path(__file__).resolve().parent.parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

warnings.filterwarnings("ignore", message=".*divide by zero.*matmul.*")
warnings.filterwarnings("ignore", message=".*overflow.*matmul.*")
warnings.filterwarnings("ignore", message=".*invalid value.*matmul.*")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess import load_posts, preprocess
from src.embed import load_or_compute_embeddings, calculate_readiness_score
from src.visualize import generate_embedding_plots
from src.evaluate import truncate_mrl, generate_comparison_table
from src.similarity import find_top_k
from src.cluster import run_clustering, run_dbscan_grid_search
from src.routing import embed_label_prototypes, route_posts
from src.hierarchical import cluster_per_topic
from src.evaluate import (
    compute_metrics,
    compute_per_topic_metrics,
    compute_routing_accuracy,
    compute_weak_label_metrics,
    extract_cluster_keywords,
    extract_keywords_from_texts,
    get_representative_examples,
    get_representative_examples_from_indices,
)
from src.report import (
    save_cluster_assignments,
    save_examples_markdown,
    save_hierarchy_csv,
    save_hierarchy_markdown,
    save_metrics,
    save_model_comparison_markdown,
    save_stability_report_markdown,
)
from src.stability import run_stability_experiment
from src.model_comparison import run_model_comparison


def _resolve_path(path: str) -> Path:
    """Resolve a path relative to the project root."""
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


# ===================================================================
# Shared steps: load, preprocess, embed, similarity
# ===================================================================

def _shared_steps(cfg: dict[str, Any]) -> tuple[Any, ...]:
    """Run steps 1-3 shared by both modes. Returns (df, embeddings, categories, cfg sub-dicts)."""
    data_cfg = cfg["data"]
    pre_cfg = cfg["preprocessing"]
    model_cfg = cfg["model"]
    out_cfg = cfg["output"]

    # 1. Load & preprocess
    print("\n[1] Daten laden & bereinigen")
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

    has_categories = bool(data_cfg.get("category_column") and data_cfg["category_column"] in df.columns)
    categories = df[data_cfg["category_column"]].values if has_categories else None
    if has_categories:
        print(f"  ↳ Weak labels vorhanden: {len(set(categories))} Kategorien")

    # 2. Embeddings
    print("\n[2] Embeddings erzeugen")
    gemini_cfg = model_cfg.get("gemini", {}) or {}
    embeddings, _ = load_or_compute_embeddings(
        df,
        text_col="text_clean",
        id_col=data_cfg["id_column"],
        model_name=model_cfg["name"],
        batch_size=model_cfg.get("batch_size", 32),
        embeddings_path=str(_resolve_path(out_cfg["embeddings_path"])),
        ids_path=str(_resolve_path(out_cfg["ids_path"])),
        gemini_api_key=gemini_cfg.get("api_key"),
        gemini_output_dimensionality=gemini_cfg.get("output_dimensionality"),
        gemini_task_type=gemini_cfg.get("task_type", "CLUSTERING"),
    )
    print(f"  ↳ Embedding-Shape: {embeddings.shape}")

    # 2b. Community-Readiness Score
    print("\n[2b] Community-Readiness Score")
    readiness_scores = calculate_readiness_score(
        embeddings,
        anchor_text="Ich möchte mich mit Gleichgesinnten vernetzen",
        model_name=model_cfg["name"],
        gemini_api_key=gemini_cfg.get("api_key"),
        gemini_output_dimensionality=gemini_cfg.get("output_dimensionality"),
    )
    df = df.copy()
    df["readiness_score"] = readiness_scores
    top5 = df.nlargest(5, "readiness_score")[["id", "readiness_score", "text_clean"]]
    print(f"  ↳ Ø Score: {readiness_scores.mean():.4f}  |  Max: {readiness_scores.max():.4f}")
    print("  ↳ Top-5 Posts mit höchster Community-Bereitschaft:")
    for _, row in top5.iterrows():
        print(f"     (score={row['readiness_score']:.4f})  {row['text_clean'][:90]}")

    # 3. Similarity search
    print("\n[3] Similarity Search")
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

    return df, embeddings, categories, data_cfg, out_cfg, cfg


# ===================================================================
# Flat mode (original pipeline)
# ===================================================================

def run_flat(cfg: dict[str, Any]) -> None:
    """Original flat clustering pipeline."""
    df, embeddings, categories, data_cfg, out_cfg, cfg = _shared_steps(cfg)

    print("\n[4] Flat Clustering")
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

            print(f"\n  → DBSCAN GRID SEARCH ({len(eps_values)}×{len(ms_values)} Kombi, metric={metric})")
            grid_df, best = run_dbscan_grid_search(
                embeddings, eps_values=eps_values,
                min_samples_values=ms_values, metric=metric,
            )
            grid_path = _resolve_path(out_cfg.get("dbscan_grid_path", "results/dbscan_grid.csv"))
            grid_path.parent.mkdir(parents=True, exist_ok=True)
            grid_df.to_csv(grid_path, index=False)
            print(f"    Grid → {grid_path}")
            print(f"    Best: eps={best['eps']}, min_samples={best['min_samples']}")

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
        print(f"    Cluster: {len(unique)} (Noise: {(labels == -1).sum()})")
        print(f"    Silhouette={metrics.get('silhouette')}, DBI={metrics.get('davies_bouldin')}")

        cluster_labels[name] = labels
        all_metrics[name] = metrics

        keywords = extract_cluster_keywords(df, labels, text_col="text_clean", top_n=top_kw)
        examples = get_representative_examples(
            df, embeddings, labels, text_col="text_clean",
            id_col=data_cfg["id_column"], n_examples=n_examples,
        )
        method_details: dict[str, Any] = {}
        for cid in sorted(set(labels)):
            method_details[str(cid)] = {
                "keywords": keywords.get(cid, []),
                "examples": examples.get(cid, []),
            }
        cluster_details[name] = method_details

    # Weak-label eval
    if categories is not None:
        print("\n[5] Weak-Label Evaluation")
        for name, labels in cluster_labels.items():
            wl = compute_weak_label_metrics(labels, categories)
            all_metrics[name]["weak_label"] = wl
            print(f"  → {name}: NMI={wl['nmi']}, ARI={wl['ari']}, Purity={wl['purity']}")
    else:
        print("\n[5] Weak-Label Evaluation – übersprungen")

    # Reports
    print("\n[6] Reports generieren")
    save_metrics(all_metrics, _resolve_path(out_cfg["metrics_path"]))
    save_cluster_assignments(df, cluster_labels, _resolve_path(out_cfg["clusters_path"]))
    save_examples_markdown(cluster_details, _resolve_path(out_cfg["examples_path"]))

    print("\n✅ Flat Pipeline abgeschlossen!")


# ===================================================================
# Hierarchical mode (two-stage)
# ===================================================================

def run_hierarchical(cfg: dict[str, Any]) -> None:
    """Two-stage pipeline: topic routing → per-topic subclustering."""
    df, embeddings, categories, _data_cfg, out_cfg, cfg = _shared_steps(cfg)

    hier_cfg = cfg.get("hierarchical", {})
    eval_cfg = cfg["evaluation"]
    top_kw = eval_cfg.get("top_keywords", 5)
    n_examples = eval_cfg.get("representative_examples", 3)
    all_metrics: dict[str, Any] = {}

    # --- Stage 1: Topic Routing ---
    print("\n[4] Stage 1 – Topic Routing")
    interest_labels: dict[str, list[str]] = cfg.get("interest_labels", {})
    if not interest_labels:
        raise ValueError("interest_labels in config.yaml fehlt oder ist leer")

    proto_cache = str(_resolve_path(out_cfg.get("proto_cache_dir", "results/proto_cache")))
    _gemini_cfg = cfg["model"].get("gemini", {}) or {}
    prototypes = embed_label_prototypes(
        interest_labels,
        model_name=cfg["model"]["name"],
        batch_size=cfg["model"].get("batch_size", 32),
        cache_dir=proto_cache,
        gemini_cfg=_gemini_cfg,
    )

    routing_df = route_posts(embeddings, prototypes, top_n=2)
    topic_labels = routing_df["topic_1"].values

    routing_path = _resolve_path(out_cfg.get("routing_path", "results/routing.csv"))
    routing_out = df[["id", "text_clean"]].copy()
    if categories is not None:
        routing_out["category"] = categories
    for col in routing_df.columns:
        routing_out[col] = routing_df[col].values
    routing_out.to_csv(routing_path, index=False)
    print(f"  ↳ Routing gespeichert → {routing_path}")

    for topic in sorted(set(topic_labels)):
        n = int((topic_labels == topic).sum())
        print(f"    {topic}: {n} Posts")

    # Routing accuracy
    if categories is not None:
        print("\n[4b] Routing-Evaluation")
        routing_acc = compute_routing_accuracy(routing_df, categories)
        all_metrics["stage1_routing"] = routing_acc
        print(f"  ↳ Accuracy@1={routing_acc['accuracy_at_1']}, Accuracy@2={routing_acc['accuracy_at_2']}")

        topic_vs_cat = compute_weak_label_metrics(
            np.array([hash(t) for t in topic_labels]),
            categories,
        )
        all_metrics["stage1_routing"]["weak_label_nmi"] = topic_vs_cat["nmi"]
        all_metrics["stage1_routing"]["weak_label_ari"] = topic_vs_cat["ari"]

    # --- Stage 2: Per-Topic Clustering ---
    print("\n[5] Stage 2 – Per-Topic Subclustering")
    sub_method = hier_cfg.get("subcluster_method", "hdbscan")
    sub_params = hier_cfg.get("subcluster_params", {})
    fallback_k = hier_cfg.get("kmeans_fallback_k", 3)
    min_for_hdb = hier_cfg.get("min_samples_for_hdbscan", 8)

    hierarchy_df = cluster_per_topic(
        df, embeddings, topic_labels,
        method=sub_method,
        method_params=sub_params,
        kmeans_fallback_k=fallback_k,
        min_samples_for_hdbscan=min_for_hdb,
        random_state=cfg.get("seed", 42),
    )

    for topic in sorted(hierarchy_df["topic_label"].unique()):
        trows = hierarchy_df[hierarchy_df["topic_label"] == topic]
        sc = set(trows["subcluster_id"])
        sc.discard(-1)
        noise = int((trows["subcluster_id"] == -1).sum())
        method_used = trows["method_used"].iloc[0]
        print(f"    {topic}: {len(sc)} Subcluster, {noise} Noise ({method_used})")

    # Per-topic metrics
    topic_metrics = compute_per_topic_metrics(df, embeddings, hierarchy_df)
    all_metrics["stage2_clustering"] = topic_metrics

    for topic, tm in topic_metrics.items():
        sil = tm.get("silhouette", "n/a")
        dbi = tm.get("davies_bouldin", "n/a")
        print(f"    {topic}: Sil={sil}, DBI={dbi}")

    # Build hierarchical report data
    id_to_global = {rid: i for i, rid in enumerate(df["id"])}
    topic_report_data: dict[str, dict[str, Any]] = {}

    for topic in sorted(hierarchy_df["topic_label"].unique()):
        trows = hierarchy_df[hierarchy_df["topic_label"] == topic]
        subclusters = sorted(set(trows["subcluster_id"]))
        n_real = len([s for s in subclusters if s >= 0])
        n_noise = int((trows["subcluster_id"] == -1).sum())

        sc_data: dict[str, Any] = {}
        for sc_id in subclusters:
            sc_rows = trows[trows["subcluster_id"] == sc_id]
            global_idx = np.array([id_to_global[rid] for rid in sc_rows["id"] if rid in id_to_global])
            texts = [df.iloc[gi]["text_clean"] for gi in global_idx]

            kw = extract_keywords_from_texts(texts, top_n=top_kw)
            exs = get_representative_examples_from_indices(
                df, embeddings, global_idx,
                text_col="text_clean", id_col="id", n_examples=n_examples,
            )
            sc_data[str(sc_id)] = {"keywords": kw, "examples": exs}

        topic_report_data[topic] = {
            "summary": {
                "n_posts": len(trows),
                "n_subclusters": n_real,
                "n_noise": n_noise,
                "method": trows["method_used"].iloc[0],
            },
            "subclusters": sc_data,
        }

    # --- Reports ---
    print("\n[6] Reports generieren")
    save_metrics(all_metrics, _resolve_path(out_cfg["metrics_path"]))
    hierarchy_csv_path = _resolve_path(out_cfg.get("hierarchy_path", "results/hierarchy.csv"))
    save_hierarchy_csv(hierarchy_df, routing_df, df, hierarchy_csv_path)
    examples_path = _resolve_path(out_cfg["examples_path"])
    save_hierarchy_markdown(topic_report_data, examples_path)

    print("\n✅ Hierarchical Pipeline abgeschlossen!")
    print(f"   → {out_cfg['metrics_path']}")
    print(f"   → {out_cfg.get('hierarchy_path', 'results/hierarchy.csv')}")
    print(f"   → {out_cfg.get('routing_path', 'results/routing.csv')}")
    print(f"   → {out_cfg['examples_path']}")

    return df, embeddings, categories, prototypes, out_cfg, cfg


# ===================================================================
# CLI
# ===================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="SBERT Social-Media Pipeline")
    parser.add_argument(
        "--config", "-c", type=str, default="configs/config.yaml",
        help="Pfad zur YAML-Konfigurationsdatei",
    )
    parser.add_argument(
        "--mode", "-m", type=str, default="hierarchical",
        choices=["flat", "hierarchical"],
        help="Pipeline-Modus: 'flat' (original) oder 'hierarchical' (two-stage, default)",
    )
    parser.add_argument(
        "--stability", action="store_true", default=False,
        help="Stability Benchmark nach dem hierarchical Run ausführen",
    )
    parser.add_argument(
        "--compare", action="store_true", default=False,
        help="Modellvergleich: alle Modelle aus config.models benchmarken",
    )
    parser.add_argument(
        "--visualize", action="store_true", default=False,
        help="t-SNE Plots erzeugen (topic_separation.png + readiness_heatmap.png)",
    )
    parser.add_argument(
        "--eval-compare", action="store_true", default=False,
        dest="eval_compare",
        help="Wissenschaftlicher Backend-Vergleich: SBERT vs. Gemini 3072 vs. Gemini 768 (MRL)",
    )
    args = parser.parse_args()

    cfg_file = Path(args.config)
    if not cfg_file.is_absolute():
        cfg_file = PROJECT_ROOT / cfg_file
    if not cfg_file.exists():
        raise FileNotFoundError(f"Config nicht gefunden: {cfg_file}")

    with open(cfg_file, encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f)

    seed = cfg.get("seed", 42)
    np.random.seed(seed)

    stab_cfg = cfg.get("stability", {})
    run_stab = args.stability or stab_cfg.get("enabled", False)

    # --- Model comparison mode ---
    if args.compare:
        print("Mode: MODEL COMPARISON")
        _run_compare_mode(cfg)
        return

    # --- Scientific eval-compare mode ---
    if args.eval_compare:
        print("Mode: EVAL-COMPARE (SBERT vs. Gemini 3072 vs. Gemini 768 MRL)")
        _run_eval_compare(cfg)
        return

    print(f"Mode: {args.mode.upper()}")

    if args.mode == "flat":
        run_flat(cfg)
    else:
        result = run_hierarchical(cfg)

        if run_stab and result is not None:
            df, embeddings, categories, prototypes, out_cfg, full_cfg = result
            print("\n" + "=" * 60)
            print("STABILITY BENCHMARK")
            print("=" * 60)
            results_dir = _resolve_path("results")
            summary = run_stability_experiment(
                full_cfg, df, embeddings, prototypes, categories, results_dir,
            )
            report_path = results_dir / "stability_report.md"
            save_stability_report_markdown(summary, report_path)
            print("\n✅ Stability Benchmark abgeschlossen!")
            print("   → results/stability_summary.json")
            print("   → results/stability_report.md")
            print("   → results/routing_confusion.csv")
            print("   → results/stability_per_topic.csv")
            print("   → results/stability_runs/")

        if args.visualize and result is not None:
            df, embeddings, categories, prototypes, out_cfg, full_cfg = result
            print("\n" + "=" * 60)
            print("VISUALISIERUNG")
            print("=" * 60)
            model_name = full_cfg["model"]["name"]
            topic_labels = np.load(_resolve_path("results/embeddings.npy"),
                                   allow_pickle=False)  # reload guard
            # Use topic assignments from routing.csv if available
            import pandas as _pd
            _routing_path = _resolve_path(out_cfg.get("routing_path", "results/routing.csv"))
            _routing_df = _pd.read_csv(_routing_path)
            _topic_labels = _routing_df["topic_1"].values
            _readiness = df["readiness_score"].values if "readiness_score" in df.columns else None
            if _readiness is None:
                print("  ↳ Readiness Scores werden berechnet ...")
                gemini_cfg = full_cfg["model"].get("gemini", {}) or {}
                _readiness = calculate_readiness_score(
                    embeddings,
                    model_name=model_name,
                    gemini_api_key=gemini_cfg.get("api_key"),
                    gemini_output_dimensionality=gemini_cfg.get("output_dimensionality"),
                )
            figures_dir = _resolve_path("results/figures")
            backend_label = "Gemini 3072" if "gemini" in model_name.lower() else model_name.split("/")[-1]
            generate_embedding_plots(
                embeddings=embeddings,
                topic_labels=_topic_labels,
                readiness_scores=_readiness,
                output_dir=figures_dir,
                backend_name=backend_label,
            )
            print("\n✅ Visualisierung abgeschlossen!")
            print(f"   → results/figures/topic_separation.png")
            print(f"   → results/figures/readiness_heatmap.png")


def _run_compare_mode(cfg: dict[str, Any]) -> None:
    """Preprocess data once, then run the model comparison benchmark."""
    data_cfg = cfg["data"]
    pre_cfg = cfg["preprocessing"]

    print("\n[1] Daten laden & bereinigen")
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

    has_cats = bool(data_cfg.get("category_column") and data_cfg["category_column"] in df.columns)
    categories = df[data_cfg["category_column"]].values if has_cats else None

    results_dir = _resolve_path("results")
    summary_df = run_model_comparison(cfg, df, categories, results_dir)

    report_path = results_dir / "model_comparison" / "model_comparison.md"
    save_model_comparison_markdown(summary_df, report_path)

    # Also copy summary CSV to top-level results
    summary_df.to_csv(results_dir / "model_comparison_summary.csv", index=False)

    print("\n" + "=" * 60)
    print("MODELLVERGLEICH ABGESCHLOSSEN")
    print("=" * 60)
    print("   → results/model_comparison_summary.csv")
    print("   → results/model_comparison/model_comparison.md")
    print("   → results/model_comparison/<model>/  (pro Modell)")


def _run_eval_compare(cfg: dict[str, Any]) -> None:
    """Scientific backend comparison: SBERT vs. Gemini 3072 vs. Gemini 768 (MRL).

    Strategy:
    - SBERT        : Load/compute with all-MiniLM-L6-v2 (cached in eval_compare/sbert/)
    - Gemini 3072  : Load from main results/embeddings.npy (already cached)
    - Gemini 768   : truncate_mrl(gemini_3072_embs, 768)  — NO extra API calls

    For each backend routing + clustering is re-run so metrics are comparable.
    Stability results are loaded from JSON if available.
    """
    import json as _json
    import pandas as _pd

    data_cfg = cfg["data"]
    pre_cfg = cfg["preprocessing"]
    hier_cfg = cfg.get("hierarchical", {})
    interest_labels = cfg.get("interest_labels", {})
    seed = cfg.get("seed", 42)
    results_dir = _resolve_path("results")
    eval_dir = results_dir / "eval_compare"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # --- Shared preprocessing ---
    print("\n[1] Daten laden & bereinigen")
    df = load_posts(
        path=_resolve_path(data_cfg["input_path"]),
        text_col=data_cfg["text_column"],
        id_col=data_cfg["id_column"],
    )
    df = preprocess(
        df,
        text_col=data_cfg["text_column"],
        remove_urls=pre_cfg.get("remove_urls", True),
        remove_mentions=pre_cfg.get("remove_mentions", True),
        lowercase=pre_cfg.get("lowercase", False),
        min_text_length=pre_cfg.get("min_text_length", 5),
        language_filter=data_cfg.get("language_filter"),
    )
    has_cats = bool(data_cfg.get("category_column") and data_cfg["category_column"] in df.columns)
    categories = df[data_cfg["category_column"]].values if has_cats else None
    print(f"  ↳ {len(df)} Posts nach Preprocessing")

    # Helper: run routing + clustering for a given embedding matrix
    def _run_backend(name: str, emb: np.ndarray, model_name: str,
                     proto_cache: str, gemini_cfg: dict) -> dict[str, Any]:
        print(f"\n  --- Backend: {name} ({emb.shape[1]} Dims) ---")

        protos = embed_label_prototypes(
            interest_labels,
            model_name=model_name,
            batch_size=cfg["model"].get("batch_size", 32),
            cache_dir=proto_cache,
            gemini_cfg=gemini_cfg,
        )
        routing_df = route_posts(emb, protos, top_n=2)
        topic_labels_arr = routing_df["topic_1"].values

        sub_method = hier_cfg.get("subcluster_method", "hdbscan")
        sub_params = hier_cfg.get("subcluster_params", {})
        fallback_k = hier_cfg.get("kmeans_fallback_k", 4)
        min_for_hdb = hier_cfg.get("min_samples_for_hdbscan", 10)

        from src.hierarchical import cluster_per_topic
        hierarchy_df = cluster_per_topic(
            df, emb, topic_labels_arr,
            method=sub_method,
            method_params=sub_params,
            kmeans_fallback_k=fallback_k,
            min_samples_for_hdbscan=min_for_hdb,
            random_state=seed,
        )

        # Readiness score
        _gcfg = gemini_cfg if "gemini" in model_name.lower() else {}
        readiness = calculate_readiness_score(
            emb,
            model_name=model_name,
            gemini_api_key=_gcfg.get("api_key"),
            gemini_output_dimensionality=_gcfg.get("output_dimensionality"),
        )

        # Load stability if available
        stab_path = results_dir / "stability_summary.json"
        stab = None
        if stab_path.exists() and "gemini" in model_name.lower():
            stab = _json.loads(stab_path.read_text())
            print(f"  ↳ Stability-Daten geladen aus {stab_path}")

        # Save visualisation per backend
        backend_fig_dir = eval_dir / name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        generate_embedding_plots(
            embeddings=emb,
            topic_labels=topic_labels_arr,
            readiness_scores=readiness,
            output_dir=backend_fig_dir,
            backend_name=name,
        )

        return {
            "name": name,
            "embeddings": emb,
            "routing_df": routing_df,
            "hierarchy_df": hierarchy_df,
            "categories": categories,
            "readiness_scores": readiness,
            "stability_summary": stab,
        }

    gemini_cfg = cfg["model"].get("gemini", {}) or {}
    gemini_model = cfg["model"]["name"]
    sbert_model = "sentence-transformers/all-MiniLM-L6-v2"

    # --- 1. SBERT ---
    print("\n[2a] SBERT Embeddings")
    sbert_cache_dir = eval_dir / "sbert"
    sbert_cache_dir.mkdir(parents=True, exist_ok=True)
    sbert_emb, _ = load_or_compute_embeddings(
        df, text_col="text_clean", id_col=data_cfg["id_column"],
        model_name=sbert_model,
        embeddings_path=str(sbert_cache_dir / "embeddings.npy"),
        ids_path=str(sbert_cache_dir / "embedding_ids.json"),
    )
    sbert_proto_cache = str(sbert_cache_dir / "proto_cache")
    backend_sbert = _run_backend("SBERT", sbert_emb, sbert_model, sbert_proto_cache, {})

    # --- 2. Gemini 3072 ---
    print("\n[2b] Gemini 3072 Embeddings (aus Cache)")
    gemini_emb, _ = load_or_compute_embeddings(
        df, text_col="text_clean", id_col=data_cfg["id_column"],
        model_name=gemini_model,
        embeddings_path=str(_resolve_path("results/embeddings.npy")),
        ids_path=str(_resolve_path("results/embedding_ids.json")),
        gemini_api_key=gemini_cfg.get("api_key"),
        gemini_output_dimensionality=gemini_cfg.get("output_dimensionality"),
        gemini_task_type=gemini_cfg.get("task_type", "CLUSTERING"),
    )
    gemini_proto_cache = str(_resolve_path("results/proto_cache"))
    backend_gemini3072 = _run_backend(
        "Gemini 3072", gemini_emb, gemini_model, gemini_proto_cache, gemini_cfg,
    )

    # --- 3. Gemini 768 MRL (truncation, no API calls) ---
    print("\n[2c] Gemini 768 (MRL-Kürzung — kein API-Call)")
    gemini768_emb = truncate_mrl(gemini_emb, target_dim=768)
    print(f"  ↳ Truncated: {gemini_emb.shape} → {gemini768_emb.shape}")
    mrl_proto_cache = str(eval_dir / "gemini768_proto_cache")
    # For MRL prototypes: truncate the full Gemini prototypes
    gemini768_cfg = {**gemini_cfg, "output_dimensionality": 768}
    backend_gemini768 = _run_backend(
        "Gemini 768 (MRL)", gemini768_emb, gemini_model, mrl_proto_cache, gemini768_cfg,
    )

    # --- Comparison table ---
    print("\n[3] Vergleichstabelle generieren")
    backends = [backend_sbert, backend_gemini3072, backend_gemini768]
    summary_df = generate_comparison_table(backends, df, output_dir=results_dir)

    print("\n" + "=" * 60)
    print("EVAL-COMPARE ABGESCHLOSSEN")
    print("=" * 60)
    print("   → results/comparison_table.md")
    print("   → results/comparison_table.csv")
    print("   → results/eval_compare/sbert/")
    print("   → results/eval_compare/gemini_3072/")
    print("   → results/eval_compare/gemini_768_mrl/")
    print()
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
