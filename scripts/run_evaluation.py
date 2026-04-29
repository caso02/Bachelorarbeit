#!/usr/bin/env python3
"""Evaluation wrapper for the extended 8-category system.

Runs:
  - Lauf 1: Reddit Evolution-AI dataset (7 evaluated categories, 8 routing targets)
  - Lauf 2: Synthetic edge-cases (irony, boundary, readiness, multi-theme)
  - Lauf 3: Ethics stress test (30 nudges)
  - Category-Gap comparison (6 vs 8 categories on travel posts)

Usage:
    python scripts/run_evaluation.py --run 1        # Reddit Evolution-AI
    python scripts/run_evaluation.py --run 2        # Edge-cases
    python scripts/run_evaluation.py --run 3        # Ethics stress
    python scripts/run_evaluation.py --run gap      # Category-gap comparison
    python scripts/run_evaluation.py --run all      # Everything
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

warnings.filterwarnings("ignore", message=".*divide by zero.*")
warnings.filterwarnings("ignore", message=".*overflow.*")
warnings.filterwarnings("ignore", message=".*invalid value.*")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env
_env_file = PROJECT_ROOT / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

import yaml
from src.embed import load_or_compute_embeddings, calculate_readiness_score
from src.evaluate import (
    truncate_mrl,
    compute_routing_accuracy,
    compute_weak_label_metrics,
    compute_per_topic_metrics,
    generate_comparison_table,
)
from src.routing import embed_label_prototypes, route_posts
from src.hierarchical import cluster_per_topic
from src.visualize import generate_embedding_plots
from src.stability import run_stability_experiment


# ---------------------------------------------------------------------------
# Category mapping: Evolution-AI category_1 → our 7 evaluated categories
# ---------------------------------------------------------------------------

CATEGORY_MAP = {
    "tech": ["programming", "software"],
    "health": ["health"],
    "finance": ["finance/money", "crypto"],
    "sports": ["sports"],
    "entertainment": ["movies", "tv_show", "video_game", "anime/manga", "music",
                      "books", "board_game", "card_game", "rpg"],
    "travel": ["travel", "geo"],
    "social": ["sex/relationships", "parenting", "social_group"],
}

# Run 1a (broad mapping) kept for documentation — see results/run1_reddit_8cat/
# Run 1b uses narrower mapping: tech=programming+software, social without politics

N_PER_CAT = 100
SEED = 42


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_evolution_ai(data_dir: Path, n_per_cat: int = N_PER_CAT) -> tuple:
    """Load stratified sample from Evolution-AI dataset.

    Returns (texts, ids, categories) as lists.
    """
    import random
    random.seed(SEED)

    info_path = data_dir / "subreddit_info.csv"
    tsv_path = data_dir / "rspct.tsv"

    # Build subreddit → our_category mapping
    sub_to_cat = {}
    with open(info_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["in_data"] != "True":
                continue
            cat1 = row["category_1"]
            sub = row["subreddit"].lower()
            for our_cat, source_cats in CATEGORY_MAP.items():
                if cat1 in source_cats:
                    sub_to_cat[sub] = our_cat
                    break

    print(f"  Mapped {len(sub_to_cat)} subreddits → {len(CATEGORY_MAP)} categories")

    # Collect posts per category
    posts_by_cat: dict[str, list] = {c: [] for c in CATEGORY_MAP}
    with open(tsv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            sub = row.get("subreddit", "").lower()
            if sub not in sub_to_cat:
                continue
            cat = sub_to_cat[sub]
            title = row.get("title", "") or ""
            body = row.get("selftext", "") or ""
            text = (title + " " + body).strip()
            # Clean <lb> tags (Evolution-AI format)
            text = text.replace("<lb>", " ").replace("  ", " ").strip()
            if len(text) < 20:
                continue
            posts_by_cat[cat].append({
                "id": row.get("id", ""),
                "text": text,
                "category": cat,
                "subreddit": sub,
            })

    # Stratified sample
    texts, ids, categories = [], [], []
    for cat in sorted(CATEGORY_MAP):
        pool = posts_by_cat[cat]
        n = min(n_per_cat, len(pool))
        sample = random.sample(pool, n)
        for p in sample:
            texts.append(p["text"])
            ids.append(p["id"])
            categories.append(cat)
        print(f"    {cat}: {n}/{len(pool)} sampled")

    return texts, ids, categories


def load_edge_cases(path: Path) -> tuple:
    """Load synthetic edge-case posts."""
    data = json.loads(path.read_text())
    texts = [p["text"] for p in data]
    ids = [p["id"] for p in data]
    categories = []
    for p in data:
        ec = p.get("expected_category", "unknown")
        if isinstance(ec, list):
            categories.append(ec[0] if ec else "unknown")
        else:
            categories.append(ec)
    return texts, ids, categories, data


# ---------------------------------------------------------------------------
# Run backend pipeline (SBERT / Gemini 3072 / Gemini 768 MRL)
# ---------------------------------------------------------------------------

def run_single_backend(
    name: str,
    embeddings: np.ndarray,
    model_name: str,
    proto_cache: str,
    gemini_cfg: dict,
    df_dict: dict,
    categories: np.ndarray | None,
    cfg: dict,
    output_dir: Path,
) -> dict:
    """Run routing + clustering for one embedding backend."""
    interest_labels = cfg["interest_labels"]
    hier_cfg = cfg.get("hierarchical", {})
    seed = cfg.get("seed", SEED)

    print(f"\n  --- {name} ({embeddings.shape[1]}d) ---")

    protos = embed_label_prototypes(
        interest_labels,
        model_name=model_name,
        batch_size=cfg["model"].get("batch_size", 32),
        cache_dir=proto_cache,
        gemini_cfg=gemini_cfg,
    )
    routing_df = route_posts(embeddings, protos, top_n=2)
    topic_labels = routing_df["topic_1"].values

    # Subclustering
    import pandas as pd
    df = pd.DataFrame(df_dict)
    hierarchy_df = cluster_per_topic(
        df, embeddings, topic_labels,
        method=hier_cfg.get("subcluster_method", "hdbscan"),
        method_params=hier_cfg.get("subcluster_params", {}),
        kmeans_fallback_k=hier_cfg.get("kmeans_fallback_k", 4),
        min_samples_for_hdbscan=hier_cfg.get("min_samples_for_hdbscan", 10),
        random_state=seed,
    )

    # Readiness
    _gcfg = gemini_cfg if "gemini" in model_name.lower() else {}
    readiness = calculate_readiness_score(
        embeddings,
        model_name=model_name,
        gemini_api_key=_gcfg.get("api_key"),
        gemini_output_dimensionality=_gcfg.get("output_dimensionality"),
    )

    # Metrics
    result = {
        "name": name,
        "embeddings": embeddings,
        "routing_df": routing_df,
        "hierarchy_df": hierarchy_df,
        "categories": categories,
        "readiness_scores": readiness,
        "stability_summary": None,
    }

    # Save routing CSV
    backend_dir = output_dir / name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    backend_dir.mkdir(parents=True, exist_ok=True)
    routing_out = df[["id", "text_clean"]].copy() if "text_clean" in df.columns else df[["id"]].copy()
    if categories is not None:
        routing_out["category"] = categories
    for col in routing_df.columns:
        routing_out[col] = routing_df[col].values
    routing_out.to_csv(backend_dir / "routing.csv", index=False)

    # Plots
    try:
        generate_embedding_plots(
            embeddings=embeddings,
            topic_labels=topic_labels,
            readiness_scores=readiness,
            output_dir=backend_dir,
            backend_name=name,
        )
    except Exception as e:
        print(f"  ⚠ Plot generation failed: {e}")

    return result


# ---------------------------------------------------------------------------
# Lauf 1: Reddit Evolution-AI, 8 Kategorien
# ---------------------------------------------------------------------------

def run_lauf1(cfg: dict, output_dir: Path):
    """Full 3-backend comparison on Evolution-AI Reddit data."""
    import pandas as pd

    print("=" * 60)
    print("LAUF 1: Reddit Evolution-AI — 8 Kategorien")
    print("=" * 60)

    data_dir = PROJECT_ROOT / "data" / "kaggle_neu"
    texts, ids, categories = load_evolution_ai(data_dir, n_per_cat=N_PER_CAT)

    # Build DataFrame matching pipeline expectations
    df = pd.DataFrame({"id": ids, "text_clean": texts, "category": categories})
    categories_arr = np.array(categories)

    gemini_cfg = cfg["model"].get("gemini", {}) or {}
    gemini_model = cfg["model"]["name"]
    sbert_model = "sentence-transformers/all-MiniLM-L6-v2"

    # --- SBERT ---
    print("\n[1] SBERT Embeddings")
    sbert_dir = output_dir / "sbert"
    sbert_dir.mkdir(parents=True, exist_ok=True)
    sbert_emb, _ = load_or_compute_embeddings(
        df, text_col="text_clean", id_col="id",
        model_name=sbert_model,
        embeddings_path=str(sbert_dir / "embeddings.npy"),
        ids_path=str(sbert_dir / "embedding_ids.json"),
    )
    backend_sbert = run_single_backend(
        "SBERT", sbert_emb, sbert_model,
        str(sbert_dir / "proto_cache"), {},
        {"id": ids, "text_clean": texts}, categories_arr, cfg, output_dir,
    )

    # --- Gemini 3072 ---
    print("\n[2] Gemini 3072 Embeddings")
    time.sleep(10)  # Rate limit courtesy
    gemini_dir = output_dir / "gemini_3072"
    gemini_dir.mkdir(parents=True, exist_ok=True)
    gemini_emb, _ = load_or_compute_embeddings(
        df, text_col="text_clean", id_col="id",
        model_name=gemini_model,
        embeddings_path=str(gemini_dir / "embeddings.npy"),
        ids_path=str(gemini_dir / "embedding_ids.json"),
        gemini_api_key=gemini_cfg.get("api_key"),
        gemini_output_dimensionality=gemini_cfg.get("output_dimensionality"),
        gemini_task_type=gemini_cfg.get("task_type", "CLUSTERING"),
    )
    backend_gemini = run_single_backend(
        "Gemini 3072", gemini_emb, gemini_model,
        str(gemini_dir / "proto_cache"), gemini_cfg,
        {"id": ids, "text_clean": texts}, categories_arr, cfg, output_dir,
    )

    # --- Gemini 768 MRL ---
    print("\n[3] Gemini 768 (MRL) — no API call")
    gemini768_emb = truncate_mrl(gemini_emb, target_dim=768)
    mrl_cfg = {**gemini_cfg, "output_dimensionality": 768}
    backend_mrl = run_single_backend(
        "Gemini 768 (MRL)", gemini768_emb, gemini_model,
        str(output_dir / "gemini_768_mrl" / "proto_cache"), mrl_cfg,
        {"id": ids, "text_clean": texts}, categories_arr, cfg, output_dir,
    )

    # --- Comparison table ---
    print("\n[4] Vergleichstabelle")
    backends = [backend_sbert, backend_gemini, backend_mrl]
    summary_df = generate_comparison_table(backends, df, output_dir=output_dir)
    print("\n" + summary_df.to_string(index=False))

    # Save metrics
    metrics = {"timestamp": datetime.now().isoformat(), "n_posts": len(texts)}
    for b in backends:
        rd = b["routing_df"]
        cats = b["categories"]
        acc = compute_routing_accuracy(rd, cats)
        wl = compute_weak_label_metrics(
            np.array([hash(t) for t in rd["topic_1"].values]), cats,
        )
        metrics[b["name"]] = {
            "accuracy_at_1": acc["accuracy_at_1"],
            "accuracy_at_2": acc["accuracy_at_2"],
            "per_category": acc.get("per_category", {}),
            "nmi": wl["nmi"],
            "ari": wl["ari"],
            "readiness_mean": float(b["readiness_scores"].mean()),
        }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\n✅ Lauf 1 abgeschlossen → {output_dir}")


# ---------------------------------------------------------------------------
# Lauf 2: Edge-Cases
# ---------------------------------------------------------------------------

def run_lauf2(cfg: dict, output_dir: Path):
    """Edge-case evaluation with per-class metrics."""
    import pandas as pd
    from scipy import stats

    print("=" * 60)
    print("LAUF 2: Synthetische Edge-Cases")
    print("=" * 60)

    edge_path = PROJECT_ROOT / "synthetic_data" / "edge_cases.json"
    texts, ids, categories, raw_data = load_edge_cases(edge_path)

    df = pd.DataFrame({"id": ids, "text_clean": texts, "category": categories})
    categories_arr = np.array(categories)

    # Use Gemini 768 MRL (best config) for edge-case evaluation
    gemini_cfg = cfg["model"].get("gemini", {}) or {}
    gemini_model = cfg["model"]["name"]

    print("\n[1] Gemini Embeddings für Edge-Cases")
    emb_dir = output_dir / "gemini_768_mrl"
    emb_dir.mkdir(parents=True, exist_ok=True)
    full_emb, _ = load_or_compute_embeddings(
        df, text_col="text_clean", id_col="id",
        model_name=gemini_model,
        embeddings_path=str(emb_dir / "embeddings.npy"),
        ids_path=str(emb_dir / "embedding_ids.json"),
        gemini_api_key=gemini_cfg.get("api_key"),
        gemini_output_dimensionality=gemini_cfg.get("output_dimensionality"),
        gemini_task_type=gemini_cfg.get("task_type", "CLUSTERING"),
    )
    emb = truncate_mrl(full_emb, target_dim=768)

    # Route all posts
    mrl_cfg = {**gemini_cfg, "output_dimensionality": 768}
    protos = embed_label_prototypes(
        cfg["interest_labels"],
        model_name=gemini_model,
        cache_dir=str(emb_dir / "proto_cache"),
        gemini_cfg=mrl_cfg,
    )
    routing_df = route_posts(emb, protos, top_n=2)

    # Readiness scores
    readiness = calculate_readiness_score(
        emb, model_name=gemini_model,
        gemini_api_key=gemini_cfg.get("api_key"),
        gemini_output_dimensionality=768,
    )

    # --- Per-class analysis ---
    results = {"timestamp": datetime.now().isoformat()}

    # Klasse A: Ironie
    print("\n[2] Klasse A: Ironie-Accuracy")
    irony_idx = [i for i, p in enumerate(raw_data) if p["post_class"] == "irony"]
    if irony_idx:
        irony_correct = sum(
            1 for i in irony_idx
            if routing_df.iloc[i]["topic_1"] == categories_arr[i]
        )
        irony_acc = irony_correct / len(irony_idx)
        print(f"  Ironie Acc@1: {irony_acc:.4f} ({irony_correct}/{len(irony_idx)})")
        # Also check: how often does it route to surface_category?
        surface_match = sum(
            1 for i in irony_idx
            if routing_df.iloc[i]["topic_1"] == raw_data[i].get("surface_category", "")
        )
        print(f"  Surface-Category-Match: {surface_match}/{len(irony_idx)}")
        results["irony"] = {
            "accuracy_at_1": round(irony_acc, 4),
            "n": len(irony_idx),
            "surface_category_match_rate": round(surface_match / len(irony_idx), 4),
        }

    # Klasse B: Boundary
    print("\n[3] Klasse B: Grenzgänger Routing-Confidence")
    boundary_idx = [i for i, p in enumerate(raw_data) if p["post_class"] == "boundary"]
    if boundary_idx:
        confidences = [
            float(routing_df.iloc[i]["score_1"]) - float(routing_df.iloc[i]["score_2"])
            for i in boundary_idx
        ]
        boundary_acc = sum(
            1 for i in boundary_idx
            if routing_df.iloc[i]["topic_1"] == categories_arr[i]
        ) / len(boundary_idx)
        mean_conf = np.mean(confidences)
        print(f"  Boundary Acc@1: {boundary_acc:.4f}")
        print(f"  Mean Confidence: {mean_conf:.4f} (Std: {np.std(confidences):.4f})")
        # Top-2 accuracy (is correct category in top 2?)
        top2_correct = sum(
            1 for i in boundary_idx
            if categories_arr[i] in [routing_df.iloc[i]["topic_1"], routing_df.iloc[i]["topic_2"]]
        )
        print(f"  Top-2 Acc: {top2_correct / len(boundary_idx):.4f}")
        results["boundary"] = {
            "accuracy_at_1": round(boundary_acc, 4),
            "accuracy_at_2": round(top2_correct / len(boundary_idx), 4),
            "mean_confidence": round(float(mean_conf), 4),
            "std_confidence": round(float(np.std(confidences)), 4),
            "n": len(boundary_idx),
        }

    # Klasse C: Readiness t-Test
    print("\n[4] Klasse C: Readiness-Validierung (t-Test)")
    high_idx = [i for i, p in enumerate(raw_data) if p["post_class"] == "explicit_connection"]
    low_idx = [i for i, p in enumerate(raw_data) if p["post_class"] == "no_connection"]
    if high_idx and low_idx:
        high_scores = readiness[high_idx]
        low_scores = readiness[low_idx]
        t_stat, p_value = stats.ttest_ind(high_scores, low_scores, alternative="greater")
        print(f"  High Readiness: mean={high_scores.mean():.4f} (n={len(high_idx)})")
        print(f"  Low Readiness:  mean={low_scores.mean():.4f} (n={len(low_idx)})")
        print(f"  t-Test: t={t_stat:.4f}, p={p_value:.6f}")
        print(f"  Signifikant (p<0.05): {'JA' if p_value < 0.05 else 'NEIN'}")
        results["readiness"] = {
            "high_mean": round(float(high_scores.mean()), 4),
            "high_std": round(float(high_scores.std()), 4),
            "low_mean": round(float(low_scores.mean()), 4),
            "low_std": round(float(low_scores.std()), 4),
            "t_statistic": round(float(t_stat), 4),
            "p_value": round(float(p_value), 6),
            "significant_005": bool(p_value < 0.05),
            "n_high": len(high_idx),
            "n_low": len(low_idx),
        }

    # Klasse E: Multi-Thema
    print("\n[5] Klasse E: Multi-Thema Top-2 Accuracy")
    multi_idx = [i for i, p in enumerate(raw_data) if p["post_class"] == "multi_theme"]
    if multi_idx:
        top2_hits = 0
        for i in multi_idx:
            expected = raw_data[i].get("expected_category", [])
            if isinstance(expected, str):
                expected = [expected]
            predicted_top2 = {routing_df.iloc[i]["topic_1"], routing_df.iloc[i]["topic_2"]}
            if any(e in predicted_top2 for e in expected):
                top2_hits += 1
        multi_acc = top2_hits / len(multi_idx)
        print(f"  Multi-Thema Top-2 Acc: {multi_acc:.4f} ({top2_hits}/{len(multi_idx)})")
        results["multi_theme"] = {
            "top2_accuracy": round(multi_acc, 4),
            "n": len(multi_idx),
        }

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "edge_case_metrics.json").write_text(json.dumps(results, indent=2))
    print(f"\n✅ Lauf 2 abgeschlossen → {output_dir}")


# ---------------------------------------------------------------------------
# Lauf 3: Ethics Stress Test
# ---------------------------------------------------------------------------

def run_lauf3(cfg: dict, output_dir: Path):
    """Run ethics stress test on 30 manipulative nudges."""
    print("=" * 60)
    print("LAUF 3: Ethics-Stress-Test")
    print("=" * 60)

    from src.ethics_agent import EthicsAgent

    nudges_path = PROJECT_ROOT / "synthetic_data" / "ethics_stress_test.json"
    nudges = json.loads(nudges_path.read_text())

    output_dir.mkdir(parents=True, exist_ok=True)
    ethics = EthicsAgent(output_dir=str(output_dir / "ethics_reviews"))

    results = []
    for nudge in nudges:
        print(f"\n  Testing: {nudge['id']} ({nudge['manipulation_type']})")
        review_input = {
            "nudge_text": nudge["nudge_text"],
            "explanation": nudge.get("explanation", ""),
        }
        try:
            review = ethics.review_nudge(review_input)
            result = {
                **nudge,
                "actual_decision": review.get("decision", "ERROR"),
                "actual_scores": review.get("scores"),
                "actual_avg_score": review.get("avg_score"),
                "reasoning": review.get("reasoning", ""),
                "matches_expected": review.get("decision") == nudge["expected_decision"],
            }
        except Exception as e:
            print(f"    ⚠ ERROR: {e}")
            result = {
                **nudge,
                "actual_decision": "ERROR",
                "error": str(e)[:200],
                "matches_expected": False,
            }
        results.append(result)
        time.sleep(10)  # Rate limit — longer pause for Gemini Flash stability

    # Summary
    by_type = {}
    for r in results:
        t = r["manipulation_type"]
        by_type.setdefault(t, {"total": 0, "correct": 0, "decisions": {}})
        by_type[t]["total"] += 1
        if r["matches_expected"]:
            by_type[t]["correct"] += 1
        d = r["actual_decision"]
        by_type[t]["decisions"][d] = by_type[t]["decisions"].get(d, 0) + 1

    print("\n" + "=" * 40)
    print("ETHICS STRESS TEST ERGEBNISSE")
    print("=" * 40)
    for t, data in sorted(by_type.items()):
        print(f"  {t}: {data['correct']}/{data['total']} korrekt — {data['decisions']}")

    (output_dir / "stress_test_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False)
    )
    (output_dir / "summary.json").write_text(
        json.dumps(by_type, indent=2, ensure_ascii=False)
    )
    print(f"\n✅ Lauf 3 abgeschlossen → {output_dir}")


# ---------------------------------------------------------------------------
# Category-Gap Comparison (6 vs 8 categories)
# ---------------------------------------------------------------------------

def run_category_gap(cfg_8: dict, output_dir: Path):
    """Compare routing of travel posts with 6 vs 8 categories."""
    import pandas as pd

    print("=" * 60)
    print("CATEGORY-GAP-VERGLEICH: 6 vs 8 Kategorien")
    print("=" * 60)

    # Load 6-category config
    cfg_6_path = PROJECT_ROOT / "configs" / "config_6cat.yaml"
    with open(cfg_6_path) as f:
        cfg_6 = yaml.safe_load(f)

    # Load 100 travel posts from Evolution-AI
    data_dir = PROJECT_ROOT / "data" / "kaggle_neu"
    import random
    random.seed(SEED)

    info = list(csv.DictReader(open(data_dir / "subreddit_info.csv")))
    travel_subs = {r["subreddit"].lower() for r in info
                   if r["category_1"] in ("travel", "geo") and r["in_data"] == "True"}

    travel_posts = []
    with open(data_dir / "rspct.tsv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row.get("subreddit", "").lower() in travel_subs:
                text = ((row.get("title", "") or "") + " " + (row.get("selftext", "") or "")).strip()
                text = text.replace("<lb>", " ").replace("  ", " ")
                if len(text) >= 20:
                    travel_posts.append({"id": row["id"], "text": text})
    sample = random.sample(travel_posts, min(100, len(travel_posts)))
    print(f"  Travel posts: {len(sample)} sampled from {len(travel_posts)} available")

    df = pd.DataFrame({"id": [p["id"] for p in sample],
                        "text_clean": [p["text"] for p in sample]})

    # Compute Gemini 768 MRL embeddings (shared for both configs)
    gemini_cfg = cfg_8["model"].get("gemini", {}) or {}
    gemini_model = cfg_8["model"]["name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    emb_full, _ = load_or_compute_embeddings(
        df, text_col="text_clean", id_col="id",
        model_name=gemini_model,
        embeddings_path=str(output_dir / "embeddings.npy"),
        ids_path=str(output_dir / "embedding_ids.json"),
        gemini_api_key=gemini_cfg.get("api_key"),
        gemini_output_dimensionality=gemini_cfg.get("output_dimensionality"),
        gemini_task_type=gemini_cfg.get("task_type", "CLUSTERING"),
    )
    emb = truncate_mrl(emb_full, 768)

    mrl_cfg = {**gemini_cfg, "output_dimensionality": 768}

    # --- 6-category routing ---
    print("\n[1] Routing mit 6 Kategorien")
    protos_6 = embed_label_prototypes(
        cfg_6["interest_labels"], model_name=gemini_model,
        cache_dir=str(output_dir / "proto_cache_6"),
        gemini_cfg=mrl_cfg,
    )
    routing_6 = route_posts(emb, protos_6, top_n=2)
    routing_6.to_csv(output_dir / "6cat_routing.csv", index=False)

    # --- 8-category routing ---
    print("\n[2] Routing mit 8 Kategorien")
    protos_8 = embed_label_prototypes(
        cfg_8["interest_labels"], model_name=gemini_model,
        cache_dir=str(output_dir / "proto_cache_8"),
        gemini_cfg=mrl_cfg,
    )
    routing_8 = route_posts(emb, protos_8, top_n=2)
    routing_8.to_csv(output_dir / "8cat_routing.csv", index=False)

    # --- Comparison ---
    print("\n[3] Vergleich")
    travel_in_6 = (routing_6["topic_1"] == "travel").sum() if "travel" in routing_6["topic_1"].values else 0
    travel_in_8 = (routing_8["topic_1"] == "travel").sum()

    dist_6 = routing_6["topic_1"].value_counts().to_dict()
    dist_8 = routing_8["topic_1"].value_counts().to_dict()

    conf_6 = (routing_6["score_1"] - routing_6["score_2"]).mean()
    conf_8 = (routing_8["score_1"] - routing_8["score_2"]).mean()

    comparison = {
        "n_travel_posts": len(sample),
        "6cat_distribution": dist_6,
        "8cat_distribution": dist_8,
        "6cat_travel_count": int(travel_in_6),
        "8cat_travel_count": int(travel_in_8),
        "6cat_mean_confidence": round(float(conf_6), 4),
        "8cat_mean_confidence": round(float(conf_8), 4),
    }

    print(f"\n  6-Kategorien: Travel-Posts korrekt geroutet: {travel_in_6}/{len(sample)}")
    print(f"  8-Kategorien: Travel-Posts korrekt geroutet: {travel_in_8}/{len(sample)}")
    print(f"  6-Kat Verteilung: {dist_6}")
    print(f"  8-Kat Verteilung: {dist_8}")
    print(f"  6-Kat Mean Confidence: {conf_6:.4f}")
    print(f"  8-Kat Mean Confidence: {conf_8:.4f}")

    (output_dir / "comparison.json").write_text(json.dumps(comparison, indent=2))
    print(f"\n✅ Category-Gap-Vergleich abgeschlossen → {output_dir}")


# ---------------------------------------------------------------------------
# Lauf 1c: Full Pipeline (Nudge + Ethics) on Run 1b data
# ---------------------------------------------------------------------------

def run_lauf1c(cfg: dict, output_dir: Path):
    """End-to-end pipeline: Subclustering + Readiness + Nudge + Ethics on Run 1b data."""
    import pandas as pd
    from src.agent_state import AgentState
    from src.community_crew import Coordinator

    print("=" * 60)
    print("LAUF 1c: Volle Pipeline (Nudge + Ethics) auf Reddit-Daten")
    print("=" * 60)

    run1b_dir = PROJECT_ROOT / "results" / "run1_reddit_8cat"
    hier_cfg = cfg.get("hierarchical", {})
    seed = cfg.get("seed", SEED)

    # --- 1. Posts laden ---
    print("\n[1] Posts aus Run 1b laden")
    routing_csv = run1b_dir / "gemini_768_mrl" / "routing.csv"
    routing_df = pd.read_csv(routing_csv)
    df = routing_df[["id", "text_clean"]].copy()
    topic_labels = routing_df["topic_1"].values
    print(f"  {len(df)} Posts, {len(set(topic_labels))} Topics")

    # --- 2. Embeddings laden + MRL ---
    print("\n[2] Embeddings laden + MRL-Truncation")
    gemini_3072 = np.load(run1b_dir / "gemini_3072" / "embeddings.npy")
    emb_768 = truncate_mrl(gemini_3072, target_dim=768)
    print(f"  {gemini_3072.shape} → {emb_768.shape}")

    # --- 3. Subclustering ---
    print("\n[3] HDBSCAN-Subclustering per Topic")
    hierarchy_df = cluster_per_topic(
        df, emb_768, topic_labels,
        method=hier_cfg.get("subcluster_method", "hdbscan"),
        method_params=hier_cfg.get("subcluster_params", {}),
        kmeans_fallback_k=hier_cfg.get("kmeans_fallback_k", 4),
        min_samples_for_hdbscan=hier_cfg.get("min_samples_for_hdbscan", 10),
        random_state=seed,
    )
    for topic in sorted(hierarchy_df["topic_label"].unique()):
        trows = hierarchy_df[hierarchy_df["topic_label"] == topic]
        subs = set(trows["subcluster_id"]) - {-1}
        noise = int((trows["subcluster_id"] == -1).sum())
        print(f"  {topic}: {len(subs)} Subcluster, {noise} Noise")

    # --- 4. Readiness ---
    print("\n[4] Community-Readiness-Scores (Gemini 768 MRL)")
    gemini_cfg = cfg["model"].get("gemini", {}) or {}
    readiness = calculate_readiness_score(
        emb_768,
        model_name="gemini-embedding-2-preview",
        gemini_api_key=gemini_cfg.get("api_key"),
        gemini_output_dimensionality=768,
    )
    print(f"  Ø Readiness: {readiness.mean():.4f}, Max: {readiness.max():.4f}")

    # Add readiness + text_clean to hierarchy
    id_to_readiness = {df.iloc[i]["id"]: float(readiness[i]) for i in range(len(df))}
    id_to_text = {df.iloc[i]["id"]: df.iloc[i]["text_clean"] for i in range(len(df))}
    hierarchy_df["readiness_score"] = hierarchy_df["id"].map(id_to_readiness)
    hierarchy_df["text_clean"] = hierarchy_df["id"].map(id_to_text)

    # --- 5. Cluster-Summaries ---
    print("\n[5] Cluster-Summaries (Top-10 nach avg_readiness)")
    clusters = []
    for (topic, sub), grp in hierarchy_df.groupby(["topic_label", "subcluster_id"]):
        if int(sub) < 0:  # Skip noise
            continue
        top5 = grp.nlargest(5, "readiness_score")[
            ["id", "text_clean", "readiness_score"]
        ].to_dict("records")
        clusters.append({
            "topic_label": topic,
            "subcluster_id": int(sub),
            "n_posts": len(grp),
            "avg_readiness": round(float(grp["readiness_score"].mean()), 4),
            "top_posts": top5,
        })

    clusters.sort(key=lambda c: c["avg_readiness"], reverse=True)
    top_clusters = clusters[:10]
    print(f"  {len(clusters)} Cluster total, Top-10 selektiert:")
    for c in top_clusters:
        print(f"    {c['topic_label']}/sub{c['subcluster_id']}: "
              f"avg_readiness={c['avg_readiness']:.4f}, n={c['n_posts']}")

    # --- 6. AgentState ---
    print("\n[6] AgentState populieren")
    posts = df.to_dict("records")
    state = AgentState(
        query="reddit_evolution_ai",
        posts=posts,
        clusters=top_clusters,
        hierarchy_df_records=hierarchy_df.to_dict("records"),
        analysis={"clusters": clusters, "n_posts": len(df), "mean_readiness": float(readiness.mean())},
        max_nudges=10,
        max_iterations=3,
    )

    # --- 7. Nudge-Generierung ---
    print("\n[7] Nudge-Generierung (Action-Agent)")
    from src.community_crew import _generate_nudges_stateful
    state = _generate_nudges_stateful(state)
    print(f"  {len(state.nudges)} Nudges generiert")

    # --- 8. Ethics-Review + Feedback-Loop ---
    print("\n[8] Ethics-Review + Feedback-Loop")
    output_dir.mkdir(parents=True, exist_ok=True)
    coordinator = Coordinator(max_revisions=3, verbose=True)
    state = coordinator._ethics_with_revision_loop(state, output_dir=output_dir)
    print(f"  Final Nudges: {len(state.final_nudges)}")

    # --- 9. Revisionstreiber ---
    print("\n[9] Revisionstreiber-Analyse")
    revision_drivers = []
    reviews_dir = output_dir / "ethics_reviews"
    if reviews_dir.exists():
        for review_file in sorted(reviews_dir.glob("review_*.json")):
            review = json.loads(review_file.read_text())
            scores = review.get("scores") or review.get("input_nudge", {}).get("scores")
            if not scores:
                continue
            decision = review.get("decision", "UNKNOWN")
            if decision in ("REVISE", "REJECTED"):
                min_criterion = min(scores, key=lambda k: scores[k])
                revision_drivers.append({
                    "file": review_file.name,
                    "decision": decision,
                    "avg_score": review.get("avg_score"),
                    "min_criterion": min_criterion,
                    "min_score": scores[min_criterion],
                    "all_scores": scores,
                })
    if revision_drivers:
        # Count which criterion is most often the lowest
        driver_counts = {}
        for d in revision_drivers:
            c = d["min_criterion"]
            driver_counts[c] = driver_counts.get(c, 0) + 1
        print(f"  {len(revision_drivers)} REVISE/REJECTED Reviews analysiert")
        print(f"  Haupt-Treiber: {sorted(driver_counts.items(), key=lambda x: -x[1])}")
    else:
        driver_counts = {}
        print("  Keine REVISE/REJECTED Reviews gefunden")

    (output_dir / "revision_drivers.json").write_text(
        json.dumps({"drivers": revision_drivers, "driver_counts": driver_counts}, indent=2, ensure_ascii=False)
    )

    # --- 10. Pipeline-Summary ---
    print("\n[10] Pipeline-Summary")
    n_approved = sum(1 for n in state.final_nudges if n.get("ethics_decision") == "APPROVED")
    n_revised = sum(1 for n in state.final_nudges if n.get("ethics_decision") == "REVISED")
    n_rejected = len(state.nudges) - n_approved - n_revised
    revisions = [n.get("revision_count", 0) for n in state.final_nudges]
    avg_revisions = sum(revisions) / len(revisions) if revisions else 0

    summary = {
        "timestamp": datetime.now().isoformat(),
        "n_posts": len(posts),
        "n_clusters_total": len(clusters),
        "n_clusters_nudged": len(top_clusters),
        "n_nudges_generated": len(state.nudges),
        "n_approved": n_approved,
        "n_revised": n_revised,
        "n_rejected": n_rejected,
        "approval_rate": round((n_approved + n_revised) / max(len(state.nudges), 1), 4),
        "avg_revisions": round(avg_revisions, 2),
        "revision_driver_counts": driver_counts,
        "cross_lingual_note": "Anchor text 'Ich möchte mich mit Gleichgesinnten vernetzen' (German) "
                              "was used with Gemini Embedding 2 on English Reddit posts. "
                              "This cross-lingual configuration is consistent with Run 1b.",
        "readiness_mean": round(float(readiness.mean()), 4),
        "readiness_max": round(float(readiness.max()), 4),
        "embedding_model": "gemini-embedding-2-preview",
        "embedding_dim": 768,
        "nudge_model": "gemini-2.5-flash",
        "ethics_model": "gemini-2.5-flash",
    }
    (output_dir / "pipeline_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )

    print(f"\n  Approved: {n_approved}, Revised: {n_revised}, Rejected: {n_rejected}")
    print(f"  Approval-Rate: {summary['approval_rate']:.0%}")
    print(f"  Ø Revisionen: {avg_revisions:.1f}")

    # --- 11. Showcase ---
    print("\n[11] Pipeline-Showcase generieren")
    showcase_lines = [
        "# Pipeline Showcase — Lauf 1c\n",
        f"Generiert: {datetime.now().isoformat()[:10]}\n",
        f"Posts: {len(posts)} | Cluster: {len(clusters)} | Nudges: {len(state.nudges)} "
        f"| Approved: {n_approved} | Revised: {n_revised} | Rejected: {n_rejected}\n",
    ]

    # Collect examples by decision type
    examples_by_type = {"APPROVED": [], "REVISED": [], "REJECTED_OR_DISCARDED": []}
    for nudge in state.final_nudges:
        d = nudge.get("ethics_decision", "UNKNOWN")
        if d == "APPROVED":
            examples_by_type["APPROVED"].append(nudge)
        elif d == "REVISED":
            examples_by_type["REVISED"].append(nudge)

    # Check for rejected (nudges NOT in final_nudges)
    final_ids = {n.get("cluster_context", {}).get("subcluster_id") for n in state.final_nudges}
    for nudge in state.nudges:
        sub_id = nudge.get("cluster_context", {}).get("subcluster_id")
        if sub_id not in final_ids and nudge.get("nudge_text"):
            examples_by_type["REJECTED_OR_DISCARDED"].append(nudge)

    for dtype, nudges_list in examples_by_type.items():
        if not nudges_list:
            continue
        example = nudges_list[0]
        ctx = example.get("cluster_context", {})
        topic = ctx.get("topic_label", "?")
        sub = ctx.get("subcluster_id", "?")
        showcase_lines.append(f"\n## Beispiel: {dtype}\n")
        showcase_lines.append(f"**Cluster**: {topic}/sub{sub} (Thema paraphrasiert: "
                              f"Subcluster innerhalb der Kategorie {topic})\n")
        showcase_lines.append(f"**Nudge-Text**: {example.get('nudge_text', 'N/A')}\n")
        showcase_lines.append(f"**Begründung**: {example.get('explanation', 'N/A')}\n")
        if example.get("revision_count"):
            showcase_lines.append(f"**Revisionen**: {example['revision_count']}\n")

        # Find matching ethics review
        for review_file in sorted((output_dir / "ethics_reviews").glob("review_*.json")):
            review = json.loads(review_file.read_text())
            if review.get("input_nudge", {}).get("nudge_text") == example.get("nudge_text"):
                scores = review.get("scores", {})
                avg = review.get("avg_score", "?")
                showcase_lines.append(f"\n**Ethics-Score**: Ø {avg}\n")
                showcase_lines.append("| Kriterium | Score |")
                showcase_lines.append("|---|---|")
                for k, v in sorted(scores.items()):
                    showcase_lines.append(f"| {k} | {v} |")
                showcase_lines.append(f"\n**Reasoning**: {review.get('reasoning', 'N/A')[:300]}...\n")
                break

    (output_dir / "pipeline_showcase.md").write_text(
        "\n".join(showcase_lines), encoding="utf-8"
    )

    # --- 12. Persist ---
    state.persist(output_dir)

    print(f"\n{'='*60}")
    print("LAUF 1c ABGESCHLOSSEN")
    print(f"{'='*60}")
    print(f"  → {output_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extended evaluation suite")
    parser.add_argument("--run", type=str, default="all",
                        choices=["1", "1c", "2", "3", "gap", "all"],
                        help="Which run to execute")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    with open(PROJECT_ROOT / args.config) as f:
        cfg = yaml.safe_load(f)

    results_base = PROJECT_ROOT / "results"

    if args.run in ("1", "all"):
        run_lauf1(cfg, results_base / "run1_reddit_8cat")

    if args.run in ("1c",):
        run_lauf1c(cfg, results_base / "run1c_reddit_pipeline")

    if args.run in ("2", "all"):
        run_lauf2(cfg, results_base / "run2_synthetic_edge")

    if args.run in ("3", "all"):
        run_lauf3(cfg, results_base / "run3_ethics_stress")

    if args.run in ("gap", "all"):
        run_category_gap(cfg, results_base / "category_gap_comparison")


if __name__ == "__main__":
    main()
