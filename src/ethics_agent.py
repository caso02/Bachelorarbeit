"""Ethics Agent — Phase 3b: Responsible-AI-Prüfinstanz für Community-Nudges.

Der EthicsAgent prüft jeden vom ActionAgent generierten Nudge gegen vier
Responsible-AI-Policies und fällt ein klares Urteil:

  APPROVED — Alle Policies erfüllt, kein Handlungsbedarf.
  REVISE   — Kleines Problem, verbesserter Text wird vorgeschlagen.
  REJECTED — Schwerer Verstoss, Nudge darf nicht ausgespielt werden.

Jedes Review wird automatisch als JSON-Datei unter results/ethics_reviews/
persistiert.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.utils import load_dotenv, parse_llm_json, get_gemini_client, call_gemini_with_retry


# ---------------------------------------------------------------------------
# System prompt (moralischer Kompass)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
Du bist ein strenger, unabhängiger KI-Ethik-Prüfer für ein Community-System.
Du prüfst automatisch generierte Vernetzungsvorschläge (Nudges) auf ethische
Verstösse, Responsible-AI-Prinzipien UND inhaltliche Qualität.

Prüf-Policies (wende ALLE an und bewerte JEDEN Punkt mit 1–10):
1. MANIPULATION (score_manipulation): Enthält der Text Druck, Dringlichkeit,
   FOMO ("Nur noch heute", "Alle anderen machen es") oder psychologische Tricks?
   10 = keinerlei Manipulation, 1 = stark manipulativ.
2. TRANSPARENZ (score_transparenz): Ist die Begründung (explanation) ehrlich,
   verständlich und tatsächlich hilfreich — oder ist sie vage/irreführend?
   10 = sehr transparent und hilfreich, 1 = komplett vage oder irreführend.
3. DISKRIMINIERUNG (score_diskriminierung): Enthält der Text Vorurteile,
   diskriminierende Sprache oder politisch hochbrisante Inhalte?
   10 = vollkommen neutral, 1 = diskriminierend.
4. AUTONOMIE (score_autonomie): Lässt der Text dem Nutzer wirklich die freie
   Wahl, oder wird subtil Druck ausgeübt?
   10 = volle Wahlfreiheit, 1 = starker Druck.
5. RELEVANZ (score_relevanz): Passt der Nudge inhaltlich zum beschriebenen
   Cluster-Thema? Sind die genannten Interessen plausibel und spezifisch?
   10 = perfekt zutreffend, 1 = komplett themenverfehlt.
6. SPEZIFITÄT (score_spezifitaet): Ist der Nudge konkret und individuell
   formuliert — oder ist er generisch und austauschbar?
   10 = sehr spezifisch, 1 = generische Standardphrase.
7. MEHRWERT (score_mehrwert): Bietet die Erklärung dem Nutzer echten Mehrwert?
   Versteht der Nutzer WARUM diese Verbindung vorgeschlagen wird?
   10 = grosser Mehrwert, 1 = nichtssagend.

Bewertungsregeln:
- Berechne den Durchschnitt aller 7 Scores → "avg_score".
- avg_score < 5.0 → "REJECTED" (schwere Mängel)
- avg_score 5.0–7.0 → "REVISE" (Verbesserungsbedarf, schlage besseren Text vor)
- avg_score > 7.0 → "APPROVED" (qualitativ hochwertig)

SEI STRENG: Ein generischer Nudge wie "Du interessierst dich für X, vernetze
dich mit anderen" ohne konkrete Details verdient KEINE hohen Scores bei
SPEZIFITÄT und MEHRWERT. Fordere spezifische Bezüge zu den Top-Posts.

Antworte ausschliesslich im JSON-Format (kein Markdown, kein Codeblock):
{
  "decision": "APPROVED"|"REVISE"|"REJECTED",
  "reasoning": "...",
  "modified_text": null,
  "scores": {
    "manipulation": 0, "transparenz": 0, "diskriminierung": 0,
    "autonomie": 0, "relevanz": 0, "spezifitaet": 0, "mehrwert": 0
  },
  "avg_score": 0.0
}

Bei "REVISE" setze modified_text auf den verbesserten Nudge-Text (String).
Bei "APPROVED" und "REJECTED" setze modified_text auf null.
"""

_VALID_DECISIONS = {"APPROVED", "REVISE", "REJECTED"}


# ---------------------------------------------------------------------------
# EthicsAgent
# ---------------------------------------------------------------------------

class EthicsAgent:
    """Reviews ActionAgent nudges against Responsible-AI policies.

    Args:
        model_name: Gemini model to use for the ethics review.
        api_key: Google API key. Falls back to GOOGLE_API_KEY env var.
        output_dir: Directory where review JSON files are persisted.
    """

    def __init__(
        self,
        model_name: str = "gemini-flash-latest",
        api_key: Optional[str] = None,
        output_dir: str | Path = "results/ethics_reviews",
    ) -> None:
        self.model_name = model_name
        self.output_dir = Path(output_dir)

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

    def review_nudge(self, nudge: dict) -> dict:
        """Review a nudge dict produced by ActionAgent.

        Args:
            nudge: Output dict from ``ActionAgent.generate_community_nudge()``.
                   Must contain at least ``nudge_text`` and ``explanation``.

        Returns:
            Dict with keys:
                - ``decision``: "APPROVED", "REVISE", or "REJECTED".
                - ``reasoning``: Internal justification from the LLM.
                - ``modified_text``: Improved nudge text (only for "REVISE"),
                  otherwise ``None``.
        """
        nudge_text = nudge.get("nudge_text")
        if not nudge_text:
            result = {
                "decision": "REJECTED",
                "reasoning": "Leerer oder fehlender nudge_text — nichts zu prüfen.",
                "modified_text": None,
            }
            self._persist(nudge, result)
            return result

        print(f"  ↳ EthicsAgent prüft Nudge (Modell: {self.model_name}) ...")

        prompt = self._build_prompt(nudge_text, nudge.get("explanation", ""), nudge=nudge)
        raw = self._call_gemini(prompt)
        parsed = self._parse_json(raw)

        # Normalise decision to uppercase; unknown values → REVISE (safe default)
        decision = parsed.get("decision", "REVISE").strip().upper()
        if decision not in _VALID_DECISIONS:
            decision = "REVISE"

        # Score-basierte Entscheidung (überschreibt LLM-Urteil wenn vorhanden)
        avg_score = parsed.get("avg_score")
        if avg_score is not None:
            try:
                avg_score = float(avg_score)
                if avg_score < 5.0:
                    decision = "REJECTED"
                elif avg_score <= 7.0:
                    decision = "REVISE"
                else:
                    decision = "APPROVED"
            except (ValueError, TypeError):
                pass  # Behalte LLM-Entscheidung

        result = {
            "decision": decision,
            "reasoning": parsed.get("reasoning", ""),
            "modified_text": parsed.get("modified_text") if decision == "REVISE" else None,
            "scores": parsed.get("scores"),
            "avg_score": avg_score,
        }

        self._persist(nudge, result)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, nudge_text: str, explanation: str, nudge: dict | None = None) -> str:
        lines = [
            _SYSTEM_PROMPT,
            "",
            "=== Zu prüfender Vernetzungsvorschlag ===",
            f"Nudge-Text: {nudge_text}",
            f"Begründung: {explanation}",
        ]

        # Cluster-Kontext und Top-Posts für Relevanz-Prüfung
        if nudge:
            ctx = nudge.get("cluster_context", {})
            if ctx:
                lines.append(f"\nCluster-Thema: {ctx.get('topic_label', '?')} "
                             f"(Subcluster {ctx.get('subcluster_id', '?')}, "
                             f"{ctx.get('n_posts', '?')} Posts)")

            top_posts = nudge.get("_top_posts", [])
            if top_posts:
                lines.append("\nTop-Posts im Cluster (zur Relevanz-Prüfung):")
                for i, tp in enumerate(top_posts[:5], 1):
                    lines.append(f"  {i}. \"{tp.get('text_clean', '')[:150]}\"")

        lines += [
            "",
            "Prüfe den obigen Vorschlag gemäss ALLEN 7 Policies, vergib Scores "
            "und fälle dein Urteil basierend auf dem Durchschnitt.",
        ]
        return "\n".join(lines)

    def _call_gemini(self, prompt: str) -> str:
        client = get_gemini_client(api_key=self._api_key)
        return call_gemini_with_retry(client, self.model_name, prompt)

    @staticmethod
    def _parse_json(raw: str) -> dict:
        """Extract JSON from the model response, with a plain-text fallback."""
        return parse_llm_json(
            raw,
            fallback_keys={"decision": "REVISE", "reasoning": "", "modified_text": None},
        )

    def _persist(self, input_nudge: dict, result: dict) -> None:
        """Save the review to a timestamped JSON file under output_dir."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        topic = (
            input_nudge.get("cluster_context", {}).get("topic_label", "unknown")
            if isinstance(input_nudge.get("cluster_context"), dict)
            else "unknown"
        )
        sub = (
            input_nudge.get("cluster_context", {}).get("subcluster_id", "x")
            if isinstance(input_nudge.get("cluster_context"), dict)
            else "x"
        )
        filename = f"review_{topic}_sub{sub}_{ts}.json"

        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "input_nudge": input_nudge,
            **result,
        }

        out_path = self.output_dir / filename
        out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False))
        print(f"  ↳ Review gespeichert → {out_path}")

    def cleanup_old_reviews(self, keep_latest: int = 50) -> int:
        """Remove old review files, keeping only the *keep_latest* most recent.

        Returns the number of deleted files.
        """
        if not self.output_dir.exists():
            return 0
        files = sorted(self.output_dir.glob("review_*.json"), key=lambda p: p.stat().st_mtime)
        to_delete = files[:-keep_latest] if len(files) > keep_latest else []
        for f in to_delete:
            f.unlink()
        return len(to_delete)
