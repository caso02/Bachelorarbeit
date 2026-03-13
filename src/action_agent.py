"""Action Agent — Phase 3: Ethisch gestaltete Community-Vernetzungs-Nudges.

Der ActionAgent liest Cluster-Ergebnisse aus der Analyse-Pipeline und generiert
diskrete, transparente Vernetzungsvorschläge via Gemini LLM.

Design-Prinzipien (Value-Sensitive Design):
  - Unaufdringlich: Keine FOMO-Sprache, kein Druck.
  - Transparent:    Kurze Begründung warum der Vorschlag gemacht wird.
  - Autonomieerhaltend: Nutzer entscheidet selbst; kein manipulativer Ton.
  - Privatsphäre:   Keine internen IDs oder Scores im Nudge-Text.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# System prompt (VSD-Ethik)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
Du bist ein ethisch ausgerichteter Community-Assistent.
Deine Aufgabe: Formuliere einen diskret formulierten Vernetzungsvorschlag für Nutzer,
die sich in ähnlichen Interessen-Clustern befinden.

Richtlinien (Value-Sensitive Design):
- Unaufdringlich: Kein Druck, kein FOMO. Nur eine sanfte Einladung.
- Transparent: Erkläre kurz, warum diese Verbindung vorgeschlagen wird.
- Autonomieerhaltend: Der Nutzer entscheidet selbst. Kein manipulativer Ton.
- Privatsphäre: Nenne keine internen IDs oder Scores im Nudge-Text.

Antworte ausschliesslich im JSON-Format (kein Markdown, kein Codeblock):
{"nudge_text": "...", "explanation": "..."}
"""


# ---------------------------------------------------------------------------
# ActionAgent
# ---------------------------------------------------------------------------

class ActionAgent:
    """Generates community-connection nudges for a given cluster of posts.

    Args:
        model_name: Gemini model to use for text generation.
        api_key: Google API key. Falls back to GOOGLE_API_KEY env var.
        readiness_threshold: Posts below this readiness score are excluded.
    """

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
        readiness_threshold: float = 0.4,
    ) -> None:
        self.model_name = model_name
        self.readiness_threshold = readiness_threshold

        # Load .env from project root (same pattern as run_pipeline.py)
        _load_dotenv()

        self._api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Kein Google API Key gefunden. "
                "Setze GOOGLE_API_KEY als Umgebungsvariable oder übergib api_key."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_community_nudge(
        self,
        cluster_data: dict,
        top_posts: list[dict],
    ) -> dict:
        """Generate a community-connection nudge for a cluster of posts.

        Args:
            cluster_data: Cluster metadata dict with keys:
                - ``topic_label`` (str): e.g. "entertainment"
                - ``subcluster_id`` (int): numeric sub-cluster index
                - ``n_posts`` (int): total posts in cluster
                - ``keywords`` (list[str], optional): top TF-IDF keywords
            top_posts: List of post dicts, each with:
                - ``id``: post identifier
                - ``text_clean`` (str): preprocessed post text
                - ``readiness_score`` (float): community-readiness score

        Returns:
            Dict with keys:
                - ``nudge_text``: The formulated invitation (1-2 sentences).
                - ``explanation``: Transparent rationale (VSD requirement).
                - ``target_user_ids``: IDs of eligible posts/users.
                - ``cluster_context``: Mirrored cluster_data for reproducibility.
                - ``n_eligible``: Number of posts above readiness threshold.
        """
        # Filter posts below readiness threshold
        eligible = [
            p for p in top_posts
            if float(p.get("readiness_score", 0.0)) >= self.readiness_threshold
        ]

        base_result = {
            "nudge_text": None,
            "explanation": "Keine Nutzer über der Readiness-Schwelle "
                           f"({self.readiness_threshold}).",
            "target_user_ids": [],
            "cluster_context": cluster_data,
            "n_eligible": 0,
        }

        if not eligible:
            print(
                f"  ↳ Kein Post überschreitet Readiness-Schwelle "
                f"{self.readiness_threshold} — kein Nudge generiert."
            )
            return base_result

        print(
            f"  ↳ {len(eligible)} Post(s) über Readiness-Schwelle "
            f"{self.readiness_threshold} → generiere Nudge ..."
        )

        prompt = self._build_prompt(cluster_data, eligible)
        raw_response = self._call_gemini(prompt)
        parsed = self._parse_json(raw_response)

        return {
            "nudge_text": parsed.get("nudge_text"),
            "explanation": parsed.get("explanation", ""),
            "target_user_ids": [p["id"] for p in eligible],
            "cluster_context": cluster_data,
            "n_eligible": len(eligible),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, cluster_data: dict, eligible_posts: list[dict]) -> str:
        topic = cluster_data.get("topic_label", "unbekannt")
        subcluster = cluster_data.get("subcluster_id", "?")
        n_total = cluster_data.get("n_posts", len(eligible_posts))
        keywords = cluster_data.get("keywords", [])

        lines = [
            _SYSTEM_PROMPT,
            "",
            "=== Cluster-Kontext ===",
            f"Themenbereich: {topic}",
            f"Untercluster-ID: {subcluster}",
            f"Gesamtzahl Posts im Cluster: {n_total}",
        ]
        if keywords:
            lines.append(f"Häufige Schlüsselbegriffe: {', '.join(keywords)}")

        lines += [
            "",
            "=== Ausgewählte Beiträge (hohe Community-Bereitschaft) ===",
        ]
        for i, post in enumerate(eligible_posts, 1):
            lines.append(f"Beitrag {i}: {post['text_clean']}")

        lines += [
            "",
            "Generiere jetzt den Vernetzungsvorschlag gemäss den VSD-Richtlinien.",
        ]
        return "\n".join(lines)

    def _call_gemini(self, prompt: str) -> str:
        try:
            from google import genai
        except ImportError as e:
            raise ImportError(
                "google-genai ist nicht installiert. "
                "Bitte ausführen: pip install google-genai"
            ) from e

        client = genai.Client(api_key=self._api_key)
        response = client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )
        return response.text

    @staticmethod
    def _parse_json(raw: str) -> dict:
        """Extract JSON from the model response, with a plain-text fallback."""
        # Try full parse first
        try:
            return json.loads(raw.strip())
        except json.JSONDecodeError:
            pass

        # Strip markdown code fences if present
        stripped = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`")
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

        # Extract first {...} block
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # Last resort: wrap raw text
        return {"nudge_text": raw.strip(), "explanation": ""}


# ---------------------------------------------------------------------------
# .env loader (no python-dotenv dependency)
# ---------------------------------------------------------------------------

def _load_dotenv(env_path: str | Path | None = None) -> None:
    """Load KEY=VALUE pairs from a .env file into os.environ."""
    path = Path(env_path) if env_path else Path(__file__).parent.parent / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
