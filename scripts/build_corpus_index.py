#!/usr/bin/env python3
"""Build corpus index from Evolution-AI dataset for Query Mode.

Usage:
    python scripts/build_corpus_index.py                          # Full build (~97'500 posts)
    python scripts/build_corpus_index.py --n-per-category 10      # Smoketest (~390 posts)
    python scripts/build_corpus_index.py --output data/corpus_index_smoketest/  # Custom output

Produces:
    data/corpus_index/
        embeddings.npy      — (N, 768) float32 array
        post_ids.json       — list of N post IDs
        posts.parquet       — id, text_clean, subreddit, category_1
        manifest.json       — metadata (model, dim, n_posts, categories, hash, timestamp)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

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

DATA_DIR = PROJECT_ROOT / "data" / "kaggle_neu"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "corpus_index"
MODEL_NAME = "gemini-embedding-2-preview"
MRL_DIM = 768
SEED = 42


# ---------------------------------------------------------------------------
# Text cleaning (inline from UniversalCollector._clean_text)
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Remove URLs, emojis, control chars; normalise whitespace."""
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = "".join(c for c in text if unicodedata.category(c) not in ("So", "Cs"))
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", text)
    return " ".join(text.split())


def prepare_text(title: str, selftext: str) -> str:
    """Combine title + selftext, clean <lb> tags, apply text cleaning."""
    title = (title or "").strip()
    selftext = (selftext or "").strip()
    # Replace Evolution-AI line break markers
    selftext = selftext.replace("<lb>", "\n")
    combined = f"{title}\n\n{selftext}" if selftext else title
    return clean_text(combined)


# ---------------------------------------------------------------------------
# Data loading & sampling
# ---------------------------------------------------------------------------

def load_and_sample(
    tsv_path: Path,
    info_path: Path,
    n_per_category: int = 2500,
    seed: int = SEED,
) -> "pd.DataFrame":
    """Load rspct.tsv, join with subreddit_info, stratified sample.

    Returns DataFrame with columns: id, text_clean, subreddit, category_1
    """
    import pandas as pd
    rng = np.random.RandomState(seed)

    # Load subreddit → category_1 mapping (only in_data=True)
    print("[1] Lade subreddit_info.csv ...")
    sub_to_cat = {}
    with open(info_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["in_data"] == "True":
                sub_to_cat[row["subreddit"].lower()] = row["category_1"]
    categories = sorted(set(sub_to_cat.values()))
    print(f"    {len(sub_to_cat)} Subreddits → {len(categories)} Kategorien")

    # Read TSV and collect posts per category
    print("[2] Lade rspct.tsv (1M Posts) ...")
    posts_by_cat: dict[str, list] = {c: [] for c in categories}
    n_total = 0
    n_matched = 0
    with open(tsv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            n_total += 1
            sub = (row.get("subreddit") or "").lower()
            if sub not in sub_to_cat:
                continue
            cat = sub_to_cat[sub]
            text = prepare_text(row.get("title", ""), row.get("selftext", ""))
            if len(text) < 20:
                continue
            posts_by_cat[cat].append({
                "id": row.get("id", ""),
                "text_clean": text,
                "subreddit": sub,
                "category_1": cat,
            })
            n_matched += 1
    print(f"    {n_total} total → {n_matched} matched (in_data + len≥20)")

    # Stratified sample
    print(f"[3] Stratifizierte Stichprobe ({n_per_category}/Kategorie) ...")
    all_posts = []
    for cat in categories:
        pool = posts_by_cat[cat]
        n = min(n_per_category, len(pool))
        indices = rng.choice(len(pool), size=n, replace=False)
        sample = [pool[i] for i in indices]
        all_posts.extend(sample)
        print(f"    {cat}: {n}/{len(pool)}")

    df = pd.DataFrame(all_posts)
    print(f"    Gesamt: {len(df)} Posts")
    return df


# ---------------------------------------------------------------------------
# Embedding computation with checkpointing
# ---------------------------------------------------------------------------

def compute_embeddings_chunked(
    texts: list[str],
    output_dir: Path,
    model_name: str = MODEL_NAME,
    batch_size: int = 100,
    checkpoint_interval: int = 10_000,
) -> np.ndarray:
    """Compute Gemini embeddings with resumable checkpointing.

    Saves checkpoint every checkpoint_interval posts. If interrupted,
    resumes from last checkpoint on next run.
    """
    from src.embed import _embed_with_gemini
    from src.evaluate import truncate_mrl

    n = len(texts)
    checkpoint_path = output_dir / "checkpoint.json"
    partial_path = output_dir / "partial_embeddings.npy"

    # Check for existing checkpoint
    start_idx = 0
    partial_embs = None
    if checkpoint_path.exists():
        cp = json.loads(checkpoint_path.read_text())
        if partial_path.exists():
            partial_embs = np.load(partial_path)
            expected_idx = cp.get("last_completed_index", 0) + 1
            if partial_embs.shape[0] >= expected_idx:
                start_idx = expected_idx
                print(f"  ↳ Checkpoint gefunden: Setze bei Index {start_idx} fort "
                      f"({partial_embs.shape[0]} Embeddings vorhanden)")
            else:
                print(f"  ⚠ Checkpoint korrupt (shape mismatch). Starte neu.")
                start_idx = 0
                partial_embs = None

    # Compute embeddings in chunks
    all_chunks = [] if partial_embs is None else [partial_embs[:start_idx]]
    remaining = texts[start_idx:]

    if not remaining:
        print(f"  ↳ Alle {n} Embeddings bereits berechnet.")
        full_3072 = partial_embs if partial_embs is not None else np.load(partial_path)
    else:
        print(f"  Berechne Embeddings für {len(remaining)} Texte (ab Index {start_idx}) ...")
        for chunk_start in range(0, len(remaining), batch_size):
            chunk_end = min(chunk_start + batch_size, len(remaining))
            chunk_texts = remaining[chunk_start:chunk_end]

            chunk_embs = _embed_with_gemini(
                chunk_texts,
                model_name=model_name,
                task_type="CLUSTERING",
            )
            all_chunks.append(chunk_embs)

            abs_idx = start_idx + chunk_end - 1
            done = abs_idx + 1
            print(f"    {done}/{n} ({done / n * 100:.1f}%)", end="\r")

            # Checkpoint
            if done % checkpoint_interval < batch_size and done < n:
                merged = np.vstack(all_chunks)
                np.save(partial_path, merged)
                checkpoint_path.write_text(json.dumps({"last_completed_index": abs_idx}))
                print(f"\n  💾 Checkpoint bei {done}/{n}")

        full_3072 = np.vstack(all_chunks)
        print(f"\n  ✓ {full_3072.shape[0]} Embeddings berechnet ({full_3072.shape[1]}d)")

    # MRL truncation
    print(f"  MRL-Kürzung: {full_3072.shape[1]}d → {MRL_DIM}d ...")
    emb_768 = truncate_mrl(full_3072, MRL_DIM).astype(np.float32)

    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    if partial_path.exists():
        partial_path.unlink()

    return emb_768


# ---------------------------------------------------------------------------
# Manifest & idempotency
# ---------------------------------------------------------------------------

def compute_content_hash(df: "pd.DataFrame") -> str:
    """SHA-256 hash of sorted (id, text_clean) for idempotency check."""
    h = hashlib.sha256()
    for _, row in df[["id", "text_clean"]].sort_values("id").iterrows():
        h.update(f"{row['id']}:{row['text_clean'][:100]}".encode())
    return h.hexdigest()


def build_manifest(
    output_dir: Path,
    model_name: str,
    n_posts: int,
    dim: int,
    content_hash: str,
    categories: list[str],
    seed: int,
    n_per_category: int,
) -> dict:
    """Create and save manifest.json."""
    manifest = {
        "model_name": model_name,
        "dim": dim,
        "n_posts": n_posts,
        "n_categories": len(categories),
        "categories": categories,
        "content_hash": content_hash,
        "seed": seed,
        "n_per_category": n_per_category,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": "Evolution-AI (rspct.tsv + subreddit_info.csv)",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)
    )
    return manifest


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build corpus index for Query Mode")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT),
                        help="Output directory for corpus index")
    parser.add_argument("--n-per-category", type=int, default=2500,
                        help="Posts per category_1 (default: 2500, smoketest: 10)")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Gemini API batch size")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    tsv_path = DATA_DIR / "rspct.tsv"
    info_path = DATA_DIR / "subreddit_info.csv"

    # Verify input files exist
    if not tsv_path.exists():
        print(f"✗ STOPP: {tsv_path} nicht gefunden!")
        sys.exit(1)
    if not info_path.exists():
        print(f"✗ STOPP: {info_path} nicht gefunden!")
        sys.exit(1)

    print("=" * 60)
    print(f"Corpus-Index-Builder")
    print(f"  Output: {output_dir}")
    print(f"  Posts/Kategorie: {args.n_per_category}")
    print(f"  Modell: {MODEL_NAME} → {MRL_DIM}d (MRL)")
    print("=" * 60)

    # Load & sample
    import pandas as pd
    df = load_and_sample(tsv_path, info_path, args.n_per_category, SEED)

    # Idempotency check
    content_hash = compute_content_hash(df)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing.get("content_hash") == content_hash:
            emb_path = output_dir / "embeddings.npy"
            if emb_path.exists():
                print(f"\n✓ Corpus-Index ist aktuell (Hash match). Überspringe.")
                return
    print(f"\n[4] Content-Hash: {content_hash[:16]}...")

    # Compute embeddings
    print(f"\n[5] Embeddings berechnen ({MODEL_NAME}) ...")
    texts = df["text_clean"].tolist()
    embeddings = compute_embeddings_chunked(
        texts, output_dir,
        model_name=MODEL_NAME,
        batch_size=args.batch_size,
    )

    # Save
    print(f"\n[6] Speichere Corpus-Index ...")
    np.save(output_dir / "embeddings.npy", embeddings)
    print(f"  ↳ embeddings.npy: {embeddings.shape}, {embeddings.nbytes / 1024 / 1024:.1f} MB")

    post_ids = df["id"].tolist()
    (output_dir / "post_ids.json").write_text(json.dumps(post_ids))
    print(f"  ↳ post_ids.json: {len(post_ids)} IDs")

    df.to_parquet(output_dir / "posts.parquet", index=False)
    parquet_size = (output_dir / "posts.parquet").stat().st_size / 1024 / 1024
    print(f"  ↳ posts.parquet: {len(df)} Posts, {parquet_size:.1f} MB")

    categories = sorted(df["category_1"].unique().tolist())
    manifest = build_manifest(
        output_dir, MODEL_NAME, len(df), MRL_DIM,
        content_hash, categories, SEED, args.n_per_category,
    )
    print(f"  ↳ manifest.json: {manifest['n_categories']} Kategorien")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"CORPUS-INDEX FERTIG")
    print(f"{'=' * 60}")
    print(f"  Posts:      {len(df)}")
    print(f"  Kategorien: {len(categories)}")
    print(f"  Embeddings: {embeddings.shape} ({embeddings.dtype})")
    print(f"  Pfad:       {output_dir}")
    cat_counts = df["category_1"].value_counts()
    print(f"  Top-5 Kategorien:")
    for cat, count in cat_counts.head(5).items():
        print(f"    {cat}: {count}")


if __name__ == "__main__":
    main()
