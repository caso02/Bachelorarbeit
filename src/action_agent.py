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
from pathlib import Path
from typing import Optional

from src.utils import load_dotenv, parse_llm_json, get_gemini_client, call_gemini_with_retry


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

        load_dotenv()

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

    def regenerate_nudge(
        self,
        cluster_data: dict,
        top_posts: list[dict],
        previous_nudge_text: str,
        ethics_feedback: str,
    ) -> dict:
        """Regenerate a nudge incorporating Ethics-Agent feedback.

        This method is called during the Coordinator's feedback loop when
        the Ethics-Agent has REJECTED a nudge. The prompt includes the
        previous attempt and the rejection reasoning so the LLM can produce
        an improved version.

        Args:
            cluster_data: Cluster metadata (same as in generate_community_nudge).
            top_posts: Original eligible posts for context.
            previous_nudge_text: The nudge text that was rejected.
            ethics_feedback: The Ethics-Agent's reasoning for rejection.

        Returns:
            Dict with the same structure as generate_community_nudge output.
        """
        # Use all posts (no readiness re-filtering — they were already filtered)
        eligible = top_posts or []

        if not eligible:
            return {
                "nudge_text": None,
                "explanation": "Keine Posts für Regeneration verfügbar.",
                "target_user_ids": [],
                "cluster_context": cluster_data,
                "n_eligible": 0,
            }

        print(f"  ↳ Regeneriere Nudge mit Ethics-Feedback...")

        revision_context = {
            "previous_attempt": previous_nudge_text,
            "rejection_reason": ethics_feedback,
        }
        prompt = self._build_prompt(cluster_data, eligible, revision_context=revision_context)
        raw_response = self._call_gemini(prompt)
        parsed = self._parse_json(raw_response)

        return {
            "nudge_text": parsed.get("nudge_text"),
            "explanation": parsed.get("explanation", ""),
            "target_user_ids": [p["id"] for p in eligible if "id" in p],
            "cluster_context": cluster_data,
            "n_eligible": len(eligible),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        cluster_data: dict,
        eligible_posts: list[dict],
        revision_context: dict | None = None,
    ) -> str:
        """Build the LLM prompt for nudge generation.

        Args:
            cluster_data: Cluster metadata.
            eligible_posts: Posts above readiness threshold.
            revision_context: Optional dict with ``previous_attempt`` and
                ``rejection_reason`` for the feedback-loop. When provided,
                the prompt instructs the model to improve the previous
                attempt based on the Ethics-Agent's criticism.
        """
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
            lines.append(f"Beitrag {i}: {post.get('text_clean', post.get('text', ''))}")

        # --- Revision-Context (Feedback-Loop) ---
        if revision_context:
            lines += [
                "",
                "=== REVISION — Vorheriger Versuch wurde abgelehnt ===",
                f"Abgelehnter Vorschlag: {revision_context['previous_attempt']}",
                f"Ablehnungsgrund (Ethics-Agent): {revision_context['rejection_reason']}",
                "",
                "WICHTIG: Generiere einen NEUEN Vorschlag, der die obige Kritik "
                "vollständig berücksichtigt. Vermeide dieselben Fehler. "
                "Halte dich strikt an die VSD-Richtlinien.",
            ]
        else:
            lines += [
                "",
                "Generiere jetzt den Vernetzungsvorschlag gemäss den VSD-Richtlinien.",
            ]

        return "\n".join(lines)

    def _call_gemini(self, prompt: str) -> str:
        client = get_gemini_client(api_key=self._api_key)
        return call_gemini_with_retry(client, self.model_name, prompt)

    @staticmethod
    def _parse_json(raw: str) -> dict:
        """Extract JSON from the model response, with a plain-text fallback."""
        return parse_llm_json(raw, fallback_keys={"nudge_text": "", "explanation": ""})

