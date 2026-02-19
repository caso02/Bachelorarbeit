"""Sentence-BERT embedding generation with caching."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def load_or_compute_embeddings(
    df: pd.DataFrame,
    text_col: str = "text_clean",
    id_col: str = "id",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 32,
    embeddings_path: Optional[str] = None,
    ids_path: Optional[str] = None,
) -> tuple[np.ndarray, list[int | str]]:
    """Compute embeddings or load from cache if files exist and IDs match.

    Returns:
        Tuple of (embeddings array [N, D], list of post IDs).
    """
    ids = df[id_col].tolist()

    if embeddings_path and ids_path:
        emb_file = Path(embeddings_path)
        id_file = Path(ids_path)
        if emb_file.exists() and id_file.exists():
            cached_ids = json.loads(id_file.read_text())
            if cached_ids == ids:
                print(f"  ↳ Embeddings aus Cache geladen ({emb_file})")
                return np.load(emb_file), ids
            print("  ↳ Cache-IDs stimmen nicht überein – berechne neu")

    texts = df[text_col].tolist()
    if not texts:
        raise ValueError("Keine Texte zum Einbetten vorhanden")

    print(f"  ↳ Lade Modell: {model_name}")
    model = SentenceTransformer(model_name)
    print(f"  ↳ Berechne Embeddings für {len(texts)} Texte (batch_size={batch_size})")
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype=np.float64)

    if embeddings_path and ids_path:
        emb_file = Path(embeddings_path)
        id_file = Path(ids_path)
        emb_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(emb_file, embeddings)
        id_file.write_text(json.dumps(ids))
        print(f"  ↳ Embeddings gespeichert → {emb_file}")

    return embeddings, ids
