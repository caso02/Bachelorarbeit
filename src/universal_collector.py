"""UniversalCollector — Robuste Datenbeschaffung mit drei Fallback-Modi.

Prioritäts-Reihenfolge:
  Modus A  🟢  Reddit API   — praw + REDDIT_CLIENT_ID in .env
  Modus B  🟡  Kaggle CSV   — data/kaggle/reddit_data.csv
  Modus C  🔴  Mock-Fallback — data/raw/sample_posts_300.csv (immer vorhanden)

Privacy & Responsible AI:
  - Usernames werden mit SHA-256 + festem Salt pseudonymisiert.
  - Klarnamen werden nie gespeichert.
  - Texte werden von URLs, Emojis und Steuerzeichen bereinigt.

Ausgabeformat (konsistent über alle Modi):
  {"id": str, "text_clean": str, "category": str, "author_hash": str, "score": int}
"""

from __future__ import annotations

import hashlib
import os
import re
import unicodedata
from pathlib import Path
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Pfad-Konstanten
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MOCK_CSV     = PROJECT_ROOT / "data" / "raw"  / "sample_posts_300.csv"
KAGGLE_CSV   = PROJECT_ROOT / "data" / "kaggle" / "reddit_data.csv"


# ---------------------------------------------------------------------------
# UniversalCollector
# ---------------------------------------------------------------------------

class UniversalCollector:
    """Collect social-media posts from Reddit, Kaggle CSV, or a mock dataset.

    Args:
        config_path: Optional path to config.yaml (currently unused, reserved for
            future per-topic collection settings).
    """

    SALT = "zhaw_2026"

    def __init__(self, config_path: Optional[Path] = None) -> None:
        self.config_path = config_path
        # Load .env if not already done
        _load_dotenv(PROJECT_ROOT / ".env")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def collect(self, query: str = "", limit: int = 500) -> list[dict]:
        """Collect posts using the highest-priority available source.

        Args:
            query: Keyword to search / filter posts. Empty string = no filter.
            limit: Maximum number of posts to return (primarily for Reddit).

        Returns:
            List of post dicts with keys:
            ``id``, ``text_clean``, ``category``, ``author_hash``, ``score``.
        """
        posts = self._try_reddit(query, limit)
        if posts is None:
            posts = self._try_kaggle(query)
        if posts is None:
            posts = self._fallback_mock(query)
        return posts

    def anonymize_author(self, username: str) -> str:
        """Return a 16-char SHA-256 hex digest of SALT+username.

        Special values ("[deleted]", "AutoModerator", empty) return "".
        """
        if not username or username in ("[deleted]", "AutoModerator", "None"):
            return ""
        return hashlib.sha256(
            f"{self.SALT}{username}".encode("utf-8")
        ).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Modus A — Reddit
    # ------------------------------------------------------------------

    def _try_reddit(self, query: str, limit: int) -> list[dict] | None:
        if not os.environ.get("REDDIT_CLIENT_ID"):
            return None

        try:
            import praw
        except ImportError:
            print("  ⚠  praw nicht installiert (pip install praw) — überspringe Reddit-Modus")
            return None

        client_id     = os.environ["REDDIT_CLIENT_ID"]
        client_secret = os.environ.get("REDDIT_CLIENT_SECRET", "")
        user_agent    = os.environ.get("REDDIT_USER_AGENT", "universalcollector/1.0 (ZHAW)")

        try:
            reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent,
            )
            subreddit = reddit.subreddit("all")
            search_iter = (
                subreddit.search(query, limit=limit, sort="hot")
                if query
                else subreddit.hot(limit=limit)
            )

            posts = []
            for submission in search_iter:
                raw_text = f"{submission.title} {submission.selftext or ''}".strip()
                author   = str(getattr(submission.author, "name", "")) if submission.author else ""
                posts.append({
                    "id":          str(submission.id),
                    "text_clean":  self._clean_text(raw_text),
                    "category":    submission.subreddit.display_name,
                    "author_hash": self.anonymize_author(author),
                    "score":       int(submission.score),
                })

            print(f"  🟢 Reddit API aktiv — {len(posts)} Posts geladen (query='{query}')")
            return posts if posts else None

        except Exception as exc:
            print(f"  ⚠  Reddit-Fehler: {exc} — Fallback wird versucht")
            return None

    # ------------------------------------------------------------------
    # Modus B — Kaggle CSV
    # ------------------------------------------------------------------

    def _try_kaggle(self, query: str) -> list[dict] | None:
        if not KAGGLE_CSV.exists():
            return None

        try:
            df = pd.read_csv(KAGGLE_CSV)
        except Exception as exc:
            print(f"  ⚠  Kaggle-CSV Ladefehler: {exc}")
            return None

        # Flexible Spalten-Erkennung
        text_col = next(
            (c for c in ("selftext", "text", "body", "title") if c in df.columns), None
        )
        if text_col is None:
            print("  ⚠  Kaggle-CSV hat keine erkannte Text-Spalte — überspringe")
            return None

        cat_col    = next((c for c in ("subreddit", "category", "label") if c in df.columns), None)
        author_col = next((c for c in ("author", "username")             if c in df.columns), None)
        id_col     = next((c for c in ("id", "post_id", "name")          if c in df.columns), None)
        score_col  = next((c for c in ("score", "ups", "upvotes")        if c in df.columns), None)

        # Keyword-Filter
        if query:
            mask = df[text_col].astype(str).str.contains(query, case=False, na=False)
            df = df[mask]

        posts = []
        for i, row in df.iterrows():
            raw_text = str(row[text_col])
            # Merge title if available and different from text_col
            if "title" in df.columns and text_col != "title":
                raw_text = f"{row.get('title', '')} {raw_text}".strip()

            author = str(row[author_col]) if author_col else ""
            posts.append({
                "id":          str(row[id_col]) if id_col else str(i),
                "text_clean":  self._clean_text(raw_text),
                "category":    str(row[cat_col]) if cat_col else "unknown",
                "author_hash": self.anonymize_author(author),
                "score":       int(row[score_col]) if score_col else 0,
            })

        print(f"  🟡 Kaggle-Daten aktiv — {len(posts)} Posts gefiltert (query='{query}')")
        return posts if posts else None

    # ------------------------------------------------------------------
    # Modus C — Mock-Fallback
    # ------------------------------------------------------------------

    def _fallback_mock(self, query: str) -> list[dict]:
        df = pd.read_csv(MOCK_CSV)

        if query:
            mask = df["text"].astype(str).str.contains(query, case=False, na=False)
            filtered = df[mask]
            # If no matches, return all posts (never return empty)
            if filtered.empty:
                print(f"  🔴 Mock-Fallback aktiv — kein Treffer für '{query}', nutze alle 300 Posts")
                filtered = df
            else:
                df = filtered

        posts = []
        for _, row in df.iterrows():
            posts.append({
                "id":          str(row["id"]),
                "text_clean":  self._clean_text(str(row["text"])),
                "category":    str(row.get("category", "unknown")),
                "author_hash": "",
                "score":       0,
            })

        print(f"  🔴 Mock-Fallback aktiv — {len(posts)} Posts (query='{query}')")
        return posts

    # ------------------------------------------------------------------
    # Text-Bereinigung
    # ------------------------------------------------------------------

    def _clean_text(self, text: str) -> str:
        """Remove URLs, emojis, and control characters; normalise whitespace."""
        # 1. URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        # 2. Emojis + Surrogate characters (Unicode categories So, Cs)
        text = ''.join(
            c for c in text
            if unicodedata.category(c) not in ('So', 'Cs')
        )
        # 3. Control characters
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        # 4. Normalise whitespace
        return ' '.join(text.split())


# ---------------------------------------------------------------------------
# .env loader
# ---------------------------------------------------------------------------

def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key   = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
