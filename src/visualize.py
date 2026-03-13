"""Visualisation helpers for embedding analysis and report generation.

Two primary plots:
    1. Topic Separation  — 2D scatter coloured by Stage-1 topic label
    2. Readiness Heatmap — 2D scatter coloured by community-readiness score

Both accept a pre-computed 2D projection (from ``reduce_dimensions``) so the
expensive t-SNE / UMAP step is only run once per backend.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Dimensionality reduction
# ---------------------------------------------------------------------------

def reduce_dimensions(
    embeddings: np.ndarray,
    method: str = "tsne",
    n_components: int = 2,
    random_state: int = 42,
    tsne_perplexity: float = 30.0,
    tsne_n_iter: int = 1000,
) -> np.ndarray:
    """Project high-dimensional embeddings to 2D for visualisation.

    Args:
        embeddings: Array of shape (N, D).
        method: "tsne" (default, no extra dependency) or "umap" (needs umap-learn).
        n_components: Target dimensionality (almost always 2).
        random_state: Reproducibility seed.
        tsne_perplexity: t-SNE perplexity (ignored for UMAP).
        tsne_n_iter: t-SNE iterations (ignored for UMAP).

    Returns:
        Array of shape (N, n_components).
    """
    if method == "umap":
        try:
            import umap
        except ImportError as e:
            raise ImportError(
                "umap-learn ist nicht installiert. "
                "Bitte ausführen: pip install umap-learn\n"
                "Alternativ method='tsne' verwenden."
            ) from e
        reducer = umap.UMAP(n_components=n_components, random_state=random_state)
        print(f"  ↳ UMAP Projektion ({embeddings.shape[0]} × {embeddings.shape[1]} → {n_components}D) ...")
        return reducer.fit_transform(embeddings)

    # Default: t-SNE (sklearn, no extra dependency)
    from sklearn.manifold import TSNE

    print(
        f"  ↳ t-SNE Projektion ({embeddings.shape[0]} × {embeddings.shape[1]} → {n_components}D, "
        f"perplexity={tsne_perplexity}) ..."
    )
    tsne_kwargs: dict = dict(
        n_components=n_components,
        perplexity=min(tsne_perplexity, embeddings.shape[0] - 1),
        random_state=random_state,
        init="pca",
        learning_rate="auto",
    )
    # sklearn ≥1.5 renamed n_iter → max_iter; support both
    import sklearn
    _sk_version = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
    if _sk_version >= (1, 5):
        tsne_kwargs["max_iter"] = tsne_n_iter
    else:
        tsne_kwargs["n_iter"] = tsne_n_iter
    tsne = TSNE(**tsne_kwargs)
    return tsne.fit_transform(embeddings)


# ---------------------------------------------------------------------------
# Plot 1: Topic Separation
# ---------------------------------------------------------------------------

def plot_topic_separation(
    coords: np.ndarray,
    topic_labels: np.ndarray,
    output_path: str | Path,
    title: str = "Stage-1 Topic Separation",
    point_size: float = 18.0,
    alpha: float = 0.75,
    dpi: int = 300,
    figsize: tuple[float, float] = (10, 7),
) -> None:
    """Scatter plot coloured by Stage-1 topic, one colour per topic.

    Saves a high-resolution PNG ready for LaTeX inclusion.

    Args:
        coords: 2D projection array of shape (N, 2).
        topic_labels: Topic assignment per post (strings), shape (N,).
        output_path: File path for the PNG output.
        title: Plot title (appears above the figure).
        point_size: Marker size.
        alpha: Point transparency.
        dpi: Output resolution (300 recommended for publications).
        figsize: Figure size in inches (width, height).
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    topics = sorted(set(topic_labels))
    palette = sns.color_palette("tab10", n_colors=len(topics))
    color_map = {t: palette[i] for i, t in enumerate(topics)}

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#f8f9fa")

    for topic in topics:
        mask = np.array(topic_labels) == topic
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=[color_map[topic]],
            label=topic.capitalize(),
            s=point_size,
            alpha=alpha,
            edgecolors="none",
        )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Dimension 1", fontsize=10)
    ax.set_ylabel("Dimension 2", fontsize=10)
    ax.legend(
        title="Topic",
        title_fontsize=9,
        fontsize=9,
        loc="best",
        framealpha=0.9,
        markerscale=1.5,
    )
    ax.tick_params(labelsize=8)

    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  ↳ Plot gespeichert → {output_path}")


# ---------------------------------------------------------------------------
# Plot 2: Readiness Heatmap
# ---------------------------------------------------------------------------

def plot_readiness_heatmap(
    coords: np.ndarray,
    readiness_scores: np.ndarray,
    output_path: str | Path,
    topic_labels: Optional[np.ndarray] = None,
    title: str = "Community-Readiness Hotspots",
    point_size: float = 18.0,
    alpha: float = 0.85,
    dpi: int = 300,
    figsize: tuple[float, float] = (10, 7),
    cmap: str = "viridis",
) -> None:
    """Scatter plot coloured by community-readiness score (viridis heatmap).

    High-score points (bright yellow) indicate posts most semantically similar
    to the community-intent anchor sentence.  Low-score points (dark purple)
    are topically distant.

    Optionally draws faint per-topic contours for spatial orientation.

    Args:
        coords: 2D projection array of shape (N, 2).
        readiness_scores: Cosine similarity to anchor, shape (N,).
        output_path: File path for the PNG output.
        topic_labels: Optional; used to draw a faint background per-topic scatter
            for spatial orientation.
        title: Plot title.
        point_size: Marker size.
        alpha: Point transparency.
        dpi: Output resolution.
        figsize: Figure size in inches.
        cmap: Matplotlib colormap name (default "viridis").
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import seaborn as sns  # noqa: F401 — imported to apply default style

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#f8f9fa")

    # Optional faint background per-topic scatter for orientation
    if topic_labels is not None:
        ax.scatter(
            coords[:, 0], coords[:, 1],
            c="#cccccc", s=point_size * 0.6,
            alpha=0.25, edgecolors="none", zorder=1,
        )

    sc = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=readiness_scores,
        cmap=cmap,
        s=point_size,
        alpha=alpha,
        edgecolors="none",
        zorder=2,
    )

    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Community-Readiness Score", fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Dimension 1", fontsize=10)
    ax.set_ylabel("Dimension 2", fontsize=10)
    ax.tick_params(labelsize=8)

    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  ↳ Plot gespeichert → {output_path}")


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def generate_embedding_plots(
    embeddings: np.ndarray,
    topic_labels: np.ndarray,
    readiness_scores: np.ndarray,
    output_dir: str | Path,
    backend_name: str = "",
    method: str = "tsne",
    random_state: int = 42,
    dpi: int = 300,
) -> None:
    """Compute 2D projection once, then produce both plots.

    Args:
        embeddings: Full embedding matrix (N, D).
        topic_labels: Stage-1 topic assignment per post.
        readiness_scores: Community-readiness scores per post.
        output_dir: Directory where PNGs are saved.
        backend_name: Label shown in plot titles (e.g. "Gemini 3072").
        method: Projection method ("tsne" or "umap").
        random_state: Seed for reproducible projections.
        dpi: Output resolution.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label = f" — {backend_name}" if backend_name else ""
    method_tag = method.upper()

    print(f"\n  Dimensionsreduktion ({method_tag}) ...")
    coords = reduce_dimensions(embeddings, method=method, random_state=random_state)

    plot_topic_separation(
        coords=coords,
        topic_labels=topic_labels,
        output_path=output_dir / "topic_separation.png",
        title=f"Stage-1 Topic Separation{label} ({method_tag})",
        dpi=dpi,
    )

    plot_readiness_heatmap(
        coords=coords,
        readiness_scores=readiness_scores,
        topic_labels=topic_labels,
        output_path=output_dir / "readiness_heatmap.png",
        title=f"Community-Readiness Hotspots{label} ({method_tag})",
        dpi=dpi,
    )
