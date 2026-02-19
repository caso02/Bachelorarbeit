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
    path.write_text(json.dumps(all_metrics, indent=2, ensure_ascii=False, default=str))
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
    """Write a Markdown report with keywords and representative examples per cluster.

    Args:
        results: Nested dict ``{method_name: {cluster_id: {keywords, examples}}}``.
        path: Output file path.
    """
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
