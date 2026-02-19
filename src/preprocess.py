"""Text preprocessing for social media posts."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pandas as pd


_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
_MENTION_PATTERN = re.compile(r"@\w+")


def load_posts(
    path: str | Path,
    text_col: str = "text",
    id_col: str = "id",
) -> pd.DataFrame:
    """Load posts from CSV or JSON and validate required columns."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Datendatei nicht gefunden: {path}")

    if path.suffix == ".csv":
        df = pd.read_csv(path)
    elif path.suffix == ".json":
        df = pd.read_json(path)
    else:
        raise ValueError(f"Nicht unterstütztes Format: {path.suffix} (CSV oder JSON erwartet)")

    for col in (text_col, id_col):
        if col not in df.columns:
            raise KeyError(f"Spalte '{col}' fehlt in {path}")

    return df


def clean_text(
    text: str,
    remove_urls: bool = True,
    remove_mentions: bool = True,
    lowercase: bool = False,
) -> str:
    """Clean a single social-media text."""
    if not isinstance(text, str):
        return ""
    if remove_urls:
        text = _URL_PATTERN.sub("", text)
    if remove_mentions:
        text = _MENTION_PATTERN.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    if lowercase:
        text = text.lower()
    return text


def preprocess(
    df: pd.DataFrame,
    text_col: str = "text",
    remove_urls: bool = True,
    remove_mentions: bool = True,
    lowercase: bool = False,
    min_text_length: int = 5,
    language_filter: Optional[str] = None,
) -> pd.DataFrame:
    """Full preprocessing pipeline on a DataFrame of posts.

    Returns a copy with an added ``text_clean`` column and rows filtered
    by minimum length (and optionally by language).
    """
    df = df.copy()
    df["text_clean"] = df[text_col].apply(
        lambda t: clean_text(
            t,
            remove_urls=remove_urls,
            remove_mentions=remove_mentions,
            lowercase=lowercase,
        )
    )

    before = len(df)
    df = df[df["text_clean"].str.len() >= min_text_length].reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"  ↳ {dropped} Posts wegen min_text_length={min_text_length} entfernt")

    if language_filter:
        try:
            from langdetect import detect

            def _is_lang(text: str) -> bool:
                try:
                    return detect(text) == language_filter
                except Exception:
                    return False

            before = len(df)
            df = df[df["text_clean"].apply(_is_lang)].reset_index(drop=True)
            print(f"  ↳ {before - len(df)} Posts wegen Sprachfilter ({language_filter}) entfernt")
        except ImportError:
            print("  ⚠ langdetect nicht installiert – Sprachfilter übersprungen")

    return df
