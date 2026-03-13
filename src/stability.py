"""Stability benchmark: multi-run experiment for the hierarchical pipeline."""

from __future__ import annotations

import json
import random
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import StratifiedShuffleSplit

from src.evaluate import compute_routing_accuracy
from src.hierarchical import cluster_per_topic
from src.perturb import perturb_text
from src.routing import route_posts


def _stratified_subsample(
    ids: np.ndarray,
    categories: np.ndarray | None,
    ratio: float,
    seed: int,
) -> np.ndarray:
    """Return indices for a stratified subsample of ``ratio`` of the data."""
    n = len(ids)
    if categories is not None and len(set(categories)) > 1:
        sss = StratifiedShuffleSplit(n_splits=1, train_size=ratio, random_state=seed)
        train_idx, _ = next(sss.split(np.zeros(n), categories))
        return train_idx
    gen = np.random.default_rng(seed)
    k = max(1, int(n * ratio))
    return gen.choice(n, size=k, replace=False)


def run_stability_experiment(
    cfg: dict[str, Any],
    df: pd.DataFrame,
    embeddings: np.ndarray,
    prototypes: dict[str, np.ndarray],
    categories: np.ndarray | None,
    results_dir: Path,
) -> dict[str, Any]:
    """Execute multiple runs with subsampling + optional perturbation.

    Reuses pre-computed embeddings (subsets per run) for speed.

    Args:
        cfg: Full config dict.
        df: Preprocessed DataFrame.
        embeddings: Pre-computed embeddings (N, D).
        prototypes: Topic prototype vectors from Stage 1.
        categories: Ground-truth labels (or None).
        results_dir: Base directory for stability outputs.

    Returns:
        Summary dict written to stability_summary.json.
    """
    stab_cfg = cfg.get("stability", {})
    n_runs: int = stab_cfg.get("n_runs", 10)
    seeds: list[int] = stab_cfg.get("seeds", list(range(1, n_runs + 1)))
    subsample_ratio: float = stab_cfg.get("subsample_ratio", 0.8)
    do_perturb: bool = stab_cfg.get("perturb_text", False)
    perturb_prob: float = stab_cfg.get("perturb_prob", 0.15)

    hier_cfg = cfg.get("hierarchical", {})
    sub_method = hier_cfg.get("subcluster_method", "hdbscan")
    sub_params = hier_cfg.get("subcluster_params", {})
    fallback_k = hier_cfg.get("kmeans_fallback_k", 3)
    min_for_hdb = hier_cfg.get("min_samples_for_hdbscan", 8)

    runs_dir = results_dir / "stability_runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    ids = df["id"].values
    run_results: list[dict[str, Any]] = []
    run_routing_maps: list[dict[int, str]] = []
    run_subcluster_maps: list[dict[int, tuple[str, int]]] = []

    confusion_counts: dict[tuple[str, str], int] = {}

    for run_i, seed in enumerate(seeds[:n_runs]):
        print(f"\n  Run {run_i + 1}/{n_runs} (seed={seed})")

        idx = _stratified_subsample(ids, categories, subsample_ratio, seed)
        sub_ids = ids[idx]
        sub_emb = embeddings[idx]
        sub_cats = categories[idx] if categories is not None else None

        sub_df = df.iloc[idx].reset_index(drop=True)

        if do_perturb:
            perturb_rng = random.Random(seed)
            sub_df = sub_df.copy()
            sub_df["text_clean"] = sub_df["text_clean"].apply(
                lambda t, _rng=perturb_rng: perturb_text(t, prob=perturb_prob, rng=_rng)
            )

        routing_df = route_posts(sub_emb, prototypes, top_n=2)
        topic_labels = routing_df["topic_1"].values

        hierarchy_df = cluster_per_topic(
            sub_df, sub_emb, topic_labels,
            method=sub_method,
            method_params=sub_params,
            kmeans_fallback_k=fallback_k,
            min_samples_for_hdbscan=min_for_hdb,
            random_state=seed,
        )

        run_csv = runs_dir / f"run_{run_i + 1:02d}.csv"
        out = hierarchy_df[["id", "topic_label", "subcluster_id"]].copy()
        if sub_cats is not None:
            out.insert(1, "category", sub_cats[:len(out)])
        out.to_csv(run_csv, index=False)

        routing_map = dict(zip(sub_ids, topic_labels))
        subcluster_map = {
            row["id"]: (row["topic_label"], row["subcluster_id"])
            for _, row in hierarchy_df.iterrows()
        }
        run_routing_maps.append(routing_map)
        run_subcluster_maps.append(subcluster_map)

        run_info: dict[str, Any] = {"seed": seed, "n_samples": len(idx)}

        if sub_cats is not None:
            acc = compute_routing_accuracy(routing_df, sub_cats)
            run_info["accuracy_at_1"] = acc["accuracy_at_1"]
            run_info["accuracy_at_2"] = acc["accuracy_at_2"]

            for i_item in range(len(sub_cats)):
                gt = str(sub_cats[i_item]).lower()
                pred = str(topic_labels[i_item]).lower()
                key = (gt, pred)
                confusion_counts[key] = confusion_counts.get(key, 0) + 1

        run_results.append(run_info)
        print(f"    n={len(idx)}, Acc@1={run_info.get('accuracy_at_1', 'n/a')}")

    # --- Aggregate Stage 1 ---
    acc1_values = [r["accuracy_at_1"] for r in run_results if "accuracy_at_1" in r]
    acc2_values = [r["accuracy_at_2"] for r in run_results if "accuracy_at_2" in r]

    stage1_summary: dict[str, Any] = {}
    if acc1_values:
        stage1_summary = {
            "accuracy_at_1_mean": round(float(np.mean(acc1_values)), 4),
            "accuracy_at_1_std": round(float(np.std(acc1_values)), 4),
            "accuracy_at_2_mean": round(float(np.mean(acc2_values)), 4),
            "accuracy_at_2_std": round(float(np.std(acc2_values)), 4),
            "n_runs": len(acc1_values),
        }

    # Confusion matrix CSV
    if confusion_counts:
        all_cats = sorted({k[0] for k in confusion_counts} | {k[1] for k in confusion_counts})
        conf_rows: list[dict[str, Any]] = []
        for gt in all_cats:
            row: dict[str, Any] = {"category": gt}
            for pred in all_cats:
                row[pred] = confusion_counts.get((gt, pred), 0)
            conf_rows.append(row)
        conf_df = pd.DataFrame(conf_rows)
        conf_path = results_dir / "routing_confusion.csv"
        conf_df.to_csv(conf_path, index=False)
        print(f"\n  ↳ Confusion Matrix → {conf_path}")

    # --- Aggregate Stage 2: pairwise ARI per topic ---
    all_topics = sorted({t for m in run_routing_maps for t in m.values()})
    topic_ari_results: dict[str, dict[str, Any]] = {}

    for topic in all_topics:
        ari_values: list[float] = []
        n_items_used: list[int] = []

        for (i, j) in combinations(range(len(run_subcluster_maps)), 2):
            map_i = run_subcluster_maps[i]
            map_j = run_subcluster_maps[j]

            common_ids = [
                pid for pid in set(map_i) & set(map_j)
                if map_i[pid][0] == topic and map_j[pid][0] == topic
            ]
            if len(common_ids) < 4:
                continue

            labels_i = np.array([map_i[pid][1] for pid in common_ids])
            labels_j = np.array([map_j[pid][1] for pid in common_ids])

            if len(set(labels_i)) < 2 and len(set(labels_j)) < 2:
                ari_values.append(1.0)
            else:
                ari_values.append(float(adjusted_rand_score(labels_i, labels_j)))
            n_items_used.append(len(common_ids))

        if ari_values:
            topic_ari_results[topic] = {
                "mean_ari": round(float(np.mean(ari_values)), 4),
                "std_ari": round(float(np.std(ari_values)), 4),
                "n_pairs": len(ari_values),
                "mean_n_items": round(float(np.mean(n_items_used)), 1),
            }
        else:
            topic_ari_results[topic] = {
                "mean_ari": None, "std_ari": None, "n_pairs": 0, "mean_n_items": 0,
            }

    # Per-topic CSV
    topic_rows = []
    for topic, info in sorted(topic_ari_results.items()):
        topic_rows.append({"topic": topic, **info})
    pd.DataFrame(topic_rows).to_csv(results_dir / "stability_per_topic.csv", index=False)
    print(f"  ↳ Per-Topic Stability → {results_dir / 'stability_per_topic.csv'}")

    # Full summary
    summary = {
        "n_runs": n_runs,
        "subsample_ratio": subsample_ratio,
        "perturb_text": do_perturb,
        "perturb_prob": perturb_prob if do_perturb else None,
        "stage1_routing": stage1_summary,
        "stage2_subcluster_ari": topic_ari_results,
        "per_run": run_results,
    }

    summary_path = results_dir / "stability_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"  ↳ Summary → {summary_path}")

    return summary
