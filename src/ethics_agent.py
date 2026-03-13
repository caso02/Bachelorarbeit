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
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.action_agent import _load_dotenv


# ---------------------------------------------------------------------------
# System prompt (moralischer Kompass)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
Du bist ein unabhängiger KI-Ethik-Prüfer für ein Community-System.
Du prüfst automatisch generierte Vernetzungsvorschläge (Nudges) auf ethische
Verstösse und Responsible-AI-Prinzipien.

Prüf-Policies (wende ALLE an):
1. MANIPULATION: Enthält der Text Druck, Dringlichkeit, FOMO ("Nur noch heute",
   "Alle anderen machen es", "Verpasse nicht") oder psychologische Tricks?
2. TRANSPARENZ: Ist die Begründung (explanation) ehrlich, verständlich und
   tatsächlich hilfreich für den Nutzer — oder ist sie vage/irreführend?
3. DISKRIMINIERUNG/SENSIBILITÄT: Enthält der Text implizite oder explizite
   Vorurteile, diskriminierende Sprache oder politisch hochbrisante Inhalte?
4. AUTONOMIE: Lässt der Text dem Nutzer wirklich die freie Wahl, oder wird
   subtil Druck ausgeübt?

Dein Urteil:
- "APPROVED": Alle Policies erfüllt, kein Handlungsbedarf.
- "REVISE": Kleines Problem erkannt — schlage eine verbesserte Version vor.
- "REJECTED": Schwerer Verstoss — der Nudge darf nicht ausgespielt werden.

Antworte ausschliesslich im JSON-Format (kein Markdown, kein Codeblock):
{"decision": "APPROVED"|"REVISE"|"REJECTED", "reasoning": "...", "modified_text": null}

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
        model_name: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
        output_dir: str | Path = "results/ethics_reviews",
    ) -> None:
        self.model_name = model_name
        self.output_dir = Path(output_dir)

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

        prompt = self._build_prompt(nudge_text, nudge.get("explanation", ""))
        raw = self._call_gemini(prompt)
        parsed = self._parse_json(raw)

        # Normalise decision to uppercase; unknown values → REVISE (safe default)
        decision = parsed.get("decision", "REVISE").strip().upper()
        if decision not in _VALID_DECISIONS:
            decision = "REVISE"

        result = {
            "decision": decision,
            "reasoning": parsed.get("reasoning", ""),
            "modified_text": parsed.get("modified_text") if decision == "REVISE" else None,
        }

        self._persist(nudge, result)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, nudge_text: str, explanation: str) -> str:
        return "\n".join([
            _SYSTEM_PROMPT,
            "",
            "=== Zu prüfender Vernetzungsvorschlag ===",
            f"Nudge-Text: {nudge_text}",
            f"Begründung: {explanation}",
            "",
            "Prüfe den obigen Vorschlag gemäss den Policies und fälle dein Urteil.",
        ])

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
        try:
            return json.loads(raw.strip())
        except json.JSONDecodeError:
            pass

        stripped = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`")
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        return {"decision": "REVISE", "reasoning": raw.strip(), "modified_text": None}

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
