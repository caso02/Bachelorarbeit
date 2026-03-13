"""Report generation: metrics JSON, cluster CSV, examples Markdown."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def save_metrics(
    all_metrics: dict[str, dict[str, Any]],
    path: str | Path,
) -> None:
    """Write clustering metrics to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(all_metrics, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"  ↳ Metriken gespeichert → {path}")


def save_cluster_assignments(
    df: pd.DataFrame,
    cluster_columns: dict[str, Any],
    path: str | Path,
) -> None:
    """Write a CSV with post IDs, texts, and cluster labels for each method."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    out = df[["id", "text_clean"]].copy()
    if "category" in df.columns:
        out["category"] = df["category"]
    for col_name, labels in cluster_columns.items():
        out[col_name] = labels

    out.to_csv(path, index=False)
    print(f"  ↳ Cluster-Zuordnungen gespeichert → {path}")


def save_examples_markdown(
    results: dict[str, dict[str, Any]],
    path: str | Path,
) -> None:
    """Write a Markdown report with keywords and representative examples per cluster."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = ["# Cluster-Analyse – Repräsentative Beispiele\n"]

    for method, clusters in results.items():
        lines.append(f"## Methode: {method}\n")

        for cluster_id, info in sorted(clusters.items(), key=lambda x: x[0]):
            label = f"Cluster {cluster_id}" if int(cluster_id) >= 0 else "Noise"
            lines.append(f"### {label}\n")
            lines.append(f"**Top-Keywords:** {', '.join(info.get('keywords', []))}\n")
            lines.append("**Beispiele:**\n")

            for ex in info.get("examples", []):
                text = ex.get("text_clean", ex.get("text", ""))
                lines.append(f"- (ID {ex.get('id', '?')}) {text}")

            lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  ↳ Beispiel-Report gespeichert → {path}")


# ---------------------------------------------------------------------------
# Hierarchical pipeline outputs
# ---------------------------------------------------------------------------

def save_hierarchy_csv(
    hierarchy_df: pd.DataFrame,
    routing_df: pd.DataFrame,
    df: pd.DataFrame,
    path: str | Path,
) -> None:
    """Save hierarchy CSV: id, category, topic_label, topic_score, subcluster_id, method."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    out = hierarchy_df[["id", "topic_label", "subcluster_id", "method_used"]].copy()

    score_map = dict(zip(df["id"], routing_df["score_1"]))
    out["topic_score"] = out["id"].map(score_map)

    if "category" in df.columns:
        cat_map = dict(zip(df["id"], df["category"]))
        out.insert(1, "category", out["id"].map(cat_map))

    col_order = [c for c in ["id", "category", "topic_label", "topic_score",
                              "subcluster_id", "method_used"] if c in out.columns]
    out = out[col_order]
    out.to_csv(path, index=False)
    print(f"  ↳ Hierarchie-Zuordnungen gespeichert → {path}")


def save_hierarchy_markdown(
    topic_data: dict[str, dict[str, Any]],
    path: str | Path,
) -> None:
    """Write hierarchical Markdown: Topic → Subcluster with keywords + examples.

    Args:
        topic_data: Nested dict::

            {topic_name: {
                "summary": {n_posts, n_subclusters, n_noise},
                "subclusters": {
                    subcluster_id: {"keywords": [...], "examples": [...]},
                    ...
                }
            }}
        path: Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = ["# Hierarchische Cluster-Analyse (Topic → Subcluster)\n"]

    for topic, data in sorted(topic_data.items()):
        summary = data.get("summary", {})
        lines.append(f"## Topic: {topic}\n")
        lines.append(
            f"**Posts:** {summary.get('n_posts', '?')} | "
            f"**Subcluster:** {summary.get('n_subclusters', '?')} | "
            f"**Noise:** {summary.get('n_noise', 0)} | "
            f"**Methode:** {summary.get('method', '?')}\n"
        )

        subclusters = data.get("subclusters", {})
        for sc_id, sc_info in sorted(subclusters.items(), key=lambda x: int(x[0])):
            label = f"Subcluster {sc_id}" if int(sc_id) >= 0 else "Noise"
            lines.append(f"### {label}\n")
            lines.append(f"**Top-Keywords:** {', '.join(sc_info.get('keywords', []))}\n")
            lines.append("**Beispiele:**\n")

            for ex in sc_info.get("examples", []):
                text = ex.get("text_clean", ex.get("text", ""))
                lines.append(f"- (ID {ex.get('id', '?')}) {text}")

            lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  ↳ Hierarchie-Report gespeichert → {path}")


# ---------------------------------------------------------------------------
# Stability report
# ---------------------------------------------------------------------------

def save_stability_report_markdown(
    summary: dict[str, Any],
    path: str | Path,
) -> None:
    """Write a Markdown report summarising the stability benchmark results.

    Args:
        summary: Output of ``stability.run_stability_experiment()``.
        path: Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    s1 = summary.get("stage1_routing", {})
    s2 = summary.get("stage2_subcluster_ari", {})
    n_runs = summary.get("n_runs", "?")
    ratio = summary.get("subsample_ratio", "?")
    perturb = summary.get("perturb_text", False)

    lines: list[str] = [
        "# Stability Benchmark Report\n",
        f"**Runs:** {n_runs} | "
        f"**Subsample:** {ratio} | "
        f"**Perturbation:** {'ja (p=' + str(summary.get('perturb_prob', '?')) + ')' if perturb else 'nein'}\n",
        "---\n",
        "## Stage 1: Routing-Stabilität\n",
    ]

    if s1:
        lines.append(
            f"| Metrik | Wert |\n|---|---|\n"
            f"| Accuracy@1 | **{s1.get('accuracy_at_1_mean', 'n/a')}** "
            f"± {s1.get('accuracy_at_1_std', 'n/a')} |\n"
            f"| Accuracy@2 | **{s1.get('accuracy_at_2_mean', 'n/a')}** "
            f"± {s1.get('accuracy_at_2_std', 'n/a')} |\n"
        )
    else:
        lines.append("*Keine Category-Daten vorhanden – Routing-Accuracy nicht berechnet.*\n")

    lines.append(
        "Confusion Matrix: siehe `results/routing_confusion.csv`\n"
    )

    lines.append("## Stage 2: Subcluster-Stabilität (pairwise ARI)\n")

    if s2:
        lines.append("| Topic | Mean ARI | Std ARI | Pairs | Mean Items |\n")
        lines.append("|---|---|---|---|---|\n")
        for topic in sorted(s2):
            info = s2[topic]
            m_ari = info.get("mean_ari")
            s_ari = info.get("std_ari")
            lines.append(
                f"| {topic} | {m_ari if m_ari is not None else 'n/a'} "
                f"| {s_ari if s_ari is not None else 'n/a'} "
                f"| {info.get('n_pairs', 0)} "
                f"| {info.get('mean_n_items', 0)} |\n"
            )
    else:
        lines.append("*Keine Subcluster-Stabilität berechnet.*\n")

    lines.append("\n## Interpretation\n")
    lines.append(
        "- **Routing-Stabilität:** Ein Accuracy@1-Std < 0.02 zeigt, dass das "
        "Topic-Routing robust gegenüber Subsampling und Perturbation ist.\n"
        "- **Subcluster ARI:** Werte > 0.6 deuten auf stabile Subcluster-Strukturen "
        "hin. Werte < 0.3 zeigen, dass die Subgruppen stark vom Sampling abhängen "
        "und vorsichtig interpretiert werden sollten.\n"
        "- **Noise-dominierte Topics:** Wenn HDBSCAN viel Noise produziert, sinkt "
        "die Vergleichsbasis (wenige Items in echten Clustern) – der ARI wird "
        "instabiler. Ein KMeans-Fallback liefert hier stabilere, aber erzwungene "
        "Cluster.\n"
    )

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  ↳ Stability-Report gespeichert → {path}")


# ---------------------------------------------------------------------------
# Model comparison report
# ---------------------------------------------------------------------------

def save_model_comparison_markdown(
    summary_df: "pd.DataFrame",
    path: str | Path,
) -> None:
    """Write a Markdown report comparing multiple embedding models.

    Args:
        summary_df: DataFrame from ``model_comparison.run_model_comparison()``.
        path: Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        "# Modellvergleich – Sentence-BERT Embedding Models\n",
        "## Übersicht\n",
        "| Modell | Dim | Acc@1 (mean +/- std) | Mean Subcluster ARI | Runtime (s) |",
        "|---|---|---|---|---|",
    ]

    best_acc = summary_df["routing_acc1_mean"].max() if "routing_acc1_mean" in summary_df else None
    best_ari = summary_df["mean_subcluster_ari"].max() if "mean_subcluster_ari" in summary_df else None

    for _, row in summary_df.iterrows():
        acc_str = "n/a"
        if row.get("routing_acc1_mean") is not None:
            bold = "**" if row["routing_acc1_mean"] == best_acc else ""
            acc_str = f"{bold}{row['routing_acc1_mean']}{bold} +/- {row.get('routing_acc1_std', '?')}"

        ari_str = "n/a"
        if row.get("mean_subcluster_ari") is not None:
            bold = "**" if row["mean_subcluster_ari"] == best_ari else ""
            ari_str = f"{bold}{row['mean_subcluster_ari']}{bold} +/- {row.get('std_subcluster_ari', '?')}"

        short = row["model"].rsplit("/", 1)[-1]
        lines.append(
            f"| {short} | {row.get('embedding_dim', '?')} "
            f"| {acc_str} | {ari_str} "
            f"| {row.get('avg_runtime_seconds', '?')} |"
        )

    # Per-topic ARI table
    ari_cols = [c for c in summary_df.columns if c.startswith("ari_")]
    if ari_cols:
        topics = [c.replace("ari_", "") for c in ari_cols]
        lines.append("\n## Per-Topic Subcluster ARI\n")
        header = "| Modell | " + " | ".join(topics) + " |"
        sep = "|---" * (len(topics) + 1) + "|"
        lines.append(header)
        lines.append(sep)

        for _, row in summary_df.iterrows():
            short = row["model"].rsplit("/", 1)[-1]
            vals = " | ".join(
                str(row.get(c, "n/a")) for c in ari_cols
            )
            lines.append(f"| {short} | {vals} |")

    lines.append("\n## Interpretation\n")
    lines.append(
        "- **Routing-Accuracy:** Zeigt, wie gut das Modell Posts den richtigen "
        "Topics zuordnet. Höhere Accuracy = bessere semantische Trennung der Topics.\n"
        "- **Subcluster ARI:** Misst, wie stabil die feingranularen Subcluster über "
        "mehrere Runs sind. Höher = reproduzierbarere Subgruppen.\n"
        "- **Runtime:** Beinhaltet Embedding-Berechnung (gecacht nach erstem Lauf), "
        "Routing, Clustering und 10 Stability-Runs.\n"
        "- **Trade-off:** Grössere Modelle (z.B. mpnet-base mit 768 Dim) liefern oft "
        "bessere Accuracy, brauchen aber mehr Zeit und Speicher. Für Produktion "
        "kann ein kleineres Modell mit akzeptabler Accuracy bevorzugt werden.\n"
    )

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  ↳ Modellvergleich-Report gespeichert → {path}")
