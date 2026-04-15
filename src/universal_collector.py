"""UniversalCollector — Robuste Datenbeschaffung mit drei Fallback-Modi.

Prioritäts-Reihenfolge:
  Modus A  🟢  Reddit API   — praw + REDDIT_CLIENT_ID in .env
  Modus B  🟡  Kaggle CSV   — data/kaggle/*.csv (Multi-CSV-Support)
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
KAGGLE_DIR   = PROJECT_ROOT / "data" / "kaggle"


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
        from src.utils import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def collect(
        self,
        query: str = "",
        limit: int = 500,
        subreddits: list[str] | None = None,
    ) -> list[dict]:
        """Collect posts using the highest-priority available source.

        Args:
            query: Keyword to search / filter posts. Empty string = no filter.
            limit: Maximum number of posts to return (primarily for Reddit).
            subreddits: Optional list of subreddit names to filter on
                (e.g. ``["technology", "art"]``).  None = all available.

        Returns:
            List of post dicts with keys:
            ``id``, ``text_clean``, ``category``, ``author_hash``, ``score``.
        """
        posts = self._try_reddit(query, limit)
        if posts is None:
            posts = self._try_kaggle(query, limit=limit, subreddits=subreddits)
        if posts is None:
            if subreddits:
                print(f"  ⚠  Keine Daten für Subreddits {subreddits} gefunden "
                      "— kein Mock-Fallback bei expliziter Sub-Angabe")
                return []
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
    # Modus B — Kaggle CSV (Multi-CSV-Support)
    # ------------------------------------------------------------------

    def _try_kaggle(
        self,
        query: str,
        limit: int = 500,
        subreddits: list[str] | None = None,
    ) -> list[dict] | None:
        """Load posts from CSV files in data/kaggle/.

        Supports:
        - Single CSV (e.g. ``reddit_data.csv``)
        - Multiple CSVs (e.g. 50 subreddit-specific files from Kaggle)
        - Automatic column detection (body/selftext/text, subreddit/category, etc.)
        - NSFW and bot filtering (when columns are present)
        - Keyword filtering via query parameter
        - Subreddit filtering via subreddits parameter

        Returns:
            List of post dicts, or None if no CSV files found.
        """
        if not KAGGLE_DIR.exists():
            return None

        csv_files = sorted(KAGGLE_DIR.glob("*.csv"))
        if not csv_files:
            return None

        # --- Optimierung: Nur passende CSVs laden (Dateiname = Subreddit) ---
        if subreddits:
            lower_subs = [s.lower() for s in subreddits]
            csv_files = [f for f in csv_files if f.stem.lower() in lower_subs]
            if not csv_files:
                print(f"  ⚠  Keine CSV-Dateien für Subreddits {subreddits} gefunden")
                return None

        # --- Lade alle CSVs ---
        frames = []
        for csv_path in csv_files:
            try:
                df = pd.read_csv(csv_path, low_memory=False)
                if not df.empty:
                    frames.append(df)
            except Exception as exc:
                print(f"  ⚠  Kaggle-CSV Ladefehler ({csv_path.name}): {exc}")
                continue

        if not frames:
            return None

        df = pd.concat(frames, ignore_index=True)
        n_files = len(frames)

        # --- Flexible Spalten-Erkennung ---
        text_col   = next((c for c in ("body", "selftext", "text", "title") if c in df.columns), None)
        cat_col    = next((c for c in ("subreddit", "category", "label")    if c in df.columns), None)
        author_col = next((c for c in ("author", "username")               if c in df.columns), None)
        id_col     = next((c for c in ("id", "post_id", "name")            if c in df.columns), None)
        score_col  = next((c for c in ("score", "ups", "upvotes")          if c in df.columns), None)

        if text_col is None:
            print("  ⚠  Kaggle-CSVs haben keine erkannte Text-Spalte — überspringe")
            return None

        # --- Subreddit-Filter (Spalten-basiert, für Mixed-CSVs) ---
        if subreddits and cat_col and cat_col in df.columns:
            lower_subs = [s.lower() for s in subreddits]
            df = df[df[cat_col].astype(str).str.lower().isin(lower_subs)]
            if df.empty:
                print(f"  ⚠  Keine Posts für Subreddits {subreddits} nach Spalten-Filter")
                return None

        # --- Filtern: NSFW, Bots, leere Posts ---
        if "is_nsfw" in df.columns:
            df = df[df["is_nsfw"] != True]  # noqa: E712
        if "is_bot" in df.columns:
            df = df[df["is_bot"] != True]  # noqa: E712

        # Leere Texte entfernen
        text_col_orig = text_col
        df_filtered = df[df[text_col].notna() & (df[text_col].astype(str).str.strip() != "")]

        # Fallback: Wenn body/selftext komplett leer, aber title vorhanden → title nutzen
        if df_filtered.empty and text_col != "title" and "title" in df.columns:
            text_col = "title"
            df_filtered = df[df[text_col].notna() & (df[text_col].astype(str).str.strip() != "")]
            if not df_filtered.empty:
                print(f"  ℹ  '{text_col_orig}'-Spalte leer — nutze 'title' als Text ({len(df_filtered)} Posts)")

        df = df_filtered

        # --- Keyword-Filter ---
        if query:
            # Suche in Haupttext und Title (falls vorhanden)
            search_col = df[text_col].astype(str)
            if "title" in df.columns and text_col != "title":
                search_col = search_col + " " + df["title"].astype(str)
            mask = search_col.str.contains(query, case=False, na=False, regex=False)
            df = df[mask]

        if df.empty:
            return None

        # --- Limit anwenden (zufällige Stichprobe bei grossen Datasets) ---
        if len(df) > limit:
            df = df.sample(n=limit, random_state=42)

        # --- Posts bauen ---
        posts = []
        for i, row in df.iterrows():
            raw_text = str(row[text_col])
            # Title + Body zusammenführen (falls body als Haupttext, title separat)
            if "title" in df.columns and text_col != "title":
                title = str(row.get("title", ""))
                if title and title != "nan":
                    raw_text = f"{title} {raw_text}".strip()

            author = str(row[author_col]) if author_col and pd.notna(row.get(author_col)) else ""

            try:
                score = int(row[score_col]) if score_col and pd.notna(row.get(score_col)) else 0
            except (ValueError, TypeError):
                score = 0

            posts.append({
                "id":          str(row[id_col]) if id_col and pd.notna(row.get(id_col)) else str(i),
                "text_clean":  self._clean_text(raw_text),
                "category":    str(row[cat_col]) if cat_col and pd.notna(row.get(cat_col)) else "unknown",
                "author_hash": self.anonymize_author(author),
                "score":       score,
            })

        print(f"  🟡 Kaggle-Daten aktiv — {len(posts)} Posts aus {n_files} CSV(s) "
              f"(query='{query}', total verfügbar: {len(frames[0]) if len(frames) == 1 else sum(len(f) for f in frames)})")
        return posts if posts else None

    # ------------------------------------------------------------------
    # Modus C — Mock-Fallback
    # ------------------------------------------------------------------

    def _fallback_mock(self, query: str) -> list[dict]:
        df = pd.read_csv(MOCK_CSV)

        if query:
            mask = df["text"].astype(str).str.contains(query, case=False, na=False, regex=False)
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

