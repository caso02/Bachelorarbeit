"""Shared Agent State — zentraler Kommunikationskanal zwischen Agents.

Ersetzt File-I/O als primäre Inter-Agent-Kommunikation durch eine
In-Memory-Datenstruktur.  Persistierung nach JSON/CSV erfolgt erst
am Ende via ``AgentState.persist()`` (Backward-Compat mit bestehendem
Streamlit-Frontend).

Patterns:
  - Shared State:  Alle Agents lesen/schreiben denselben AgentState.
  - Agent Messaging:  Agents senden einander strukturierte Nachrichten.
  - Feedback-Loop:  Ethics → Action Revision via ``revision_queue``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Agent Message (Inter-Agent-Kommunikation)
# ---------------------------------------------------------------------------

@dataclass
class AgentMessage:
    """Eine strukturierte Nachricht von einem Agent an einen anderen.

    Messages werden im ``AgentState.messages``-Log gespeichert und ermöglichen
    eine nachvollziehbare Kommunikation zwischen Agents (Audit-Trail).
    """
    sender: str
    receiver: str
    message_type: str       # z.B. "REVISE_REQUEST", "INFO", "ERROR", "DATA_REQUEST"
    payload: dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def to_dict(self) -> dict:
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "message_type": self.message_type,
            "payload": self.payload,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Agent State (Shared State)
# ---------------------------------------------------------------------------

@dataclass
class AgentState:
    """Zentraler Shared State — ersetzt File-I/O für Inter-Agent-Kommunikation.

    Jeder Agent liest Daten aus dem State, verarbeitet sie und schreibt
    Ergebnisse zurück.  Am Ende wird ``persist()`` aufgerufen um dieselben
    JSON/CSV-Dateien zu schreiben wie bisher (Streamlit-Kompatibilität).

    Attributes:
        query: Suchbegriff / Keyword für den OODA-Loop.
        posts: Gesammelte Posts (Collector → Analyst).
        analysis: Analyse-Ergebnis inkl. Cluster-Zusammenfassung.
        clusters: Cluster-Liste sortiert nach avg_readiness.
        hierarchy_df_records: Hierarchie-DataFrame als list[dict] (für CSV-Persist).
        nudges: Generierte Nudges vor Ethics-Review.
        final_nudges: Freigegebene Nudges nach Ethics-Review.
        messages: Inter-Agent-Nachrichten (Audit-Trail).
        current_phase: Aktuelle Pipeline-Phase.
        iteration: Aktueller Revisions-Durchlauf.
        max_iterations: Maximale Revisions-Iterationen pro Nudge.
        errors: Gesammelte Fehler während der Pipeline.
    """

    # --- Daten-Flow (Pipeline) ---
    query: str = ""
    posts: list[dict] = field(default_factory=list)
    analysis: dict = field(default_factory=dict)
    clusters: list[dict] = field(default_factory=list)
    hierarchy_df_records: list[dict] = field(default_factory=list)
    nudges: list[dict] = field(default_factory=list)
    final_nudges: list[dict] = field(default_factory=list)

    # --- Konfiguration (CLI / Streamlit) ---
    limit: int = 500                                     # Max Posts
    subreddits: list[str] | None = None                  # None = alle
    max_nudges: int = 3                                  # Max Nudges

    # --- Inter-Agent-Kommunikation ---
    messages: list[AgentMessage] = field(default_factory=list)

    # --- Pipeline-Steuerung ---
    current_phase: str = "idle"
    iteration: int = 0
    max_iterations: int = 3
    errors: list[dict] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Messaging API
    # ------------------------------------------------------------------

    def add_message(
        self,
        sender: str,
        receiver: str,
        message_type: str,
        payload: dict | None = None,
    ) -> AgentMessage:
        """Fügt eine Nachricht zum Message-Log hinzu."""
        msg = AgentMessage(
            sender=sender,
            receiver=receiver,
            message_type=message_type,
            payload=payload or {},
        )
        self.messages.append(msg)
        return msg

    def get_messages_for(
        self,
        receiver: str,
        message_type: str | None = None,
    ) -> list[AgentMessage]:
        """Gibt alle Nachrichten für einen bestimmten Empfänger zurück."""
        result = [m for m in self.messages if m.receiver == receiver]
        if message_type:
            result = [m for m in result if m.message_type == message_type]
        return result

    def get_messages_between(
        self,
        sender: str,
        receiver: str,
    ) -> list[AgentMessage]:
        """Gibt alle Nachrichten zwischen zwei Agents zurück."""
        return [
            m for m in self.messages
            if m.sender == sender and m.receiver == receiver
        ]

    # ------------------------------------------------------------------
    # Serialisierung
    # ------------------------------------------------------------------

    def snapshot(self) -> dict:
        """Gibt den gesamten State als serialisierbares Dict zurück."""
        return {
            "query": self.query,
            "current_phase": self.current_phase,
            "iteration": self.iteration,
            "n_posts": len(self.posts),
            "n_clusters": len(self.clusters),
            "n_nudges": len(self.nudges),
            "n_final_nudges": len(self.final_nudges),
            "n_messages": len(self.messages),
            "n_errors": len(self.errors),
            "messages": [m.to_dict() for m in self.messages],
            "errors": self.errors,
        }

    def persist(self, results_dir: str | Path) -> None:
        """Schreibt State in dieselben JSON/CSV-Dateien wie bisher.

        Wird am Ende der Pipeline aufgerufen.  Stellt sicher, dass das
        Streamlit-Frontend weiterhin korrekt funktioniert.
        """
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        # crew_posts.json
        if self.posts:
            (results_dir / "crew_posts.json").write_text(
                json.dumps(self.posts, indent=2, ensure_ascii=False)
            )

        # crew_analysis.json
        if self.analysis:
            (results_dir / "crew_analysis.json").write_text(
                json.dumps(self.analysis, indent=2, ensure_ascii=False)
            )

        # hierarchy.csv
        if self.hierarchy_df_records:
            import pandas as pd
            pd.DataFrame(self.hierarchy_df_records).to_csv(
                results_dir / "hierarchy.csv", index=False
            )

        # crew_nudges.json
        if self.nudges:
            (results_dir / "crew_nudges.json").write_text(
                json.dumps(self.nudges, indent=2, ensure_ascii=False)
            )

        # final_nudges.json
        n_approved = sum(1 for n in self.final_nudges if n.get("ethics_decision") == "APPROVED")
        n_revised = sum(1 for n in self.final_nudges if n.get("ethics_decision") == "REVISED")
        final = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "total_reviewed": len(self.nudges),
            "approved_count": n_approved + n_revised,
            "rejected_count": len(self.nudges) - (n_approved + n_revised),
            "nudges": self.final_nudges,
        }
        (results_dir / "final_nudges.json").write_text(
            json.dumps(final, indent=2, ensure_ascii=False)
        )

        # agent_messages.json (NEU — Audit-Trail)
        if self.messages:
            (results_dir / "agent_messages.json").write_text(
                json.dumps(
                    [m.to_dict() for m in self.messages],
                    indent=2, ensure_ascii=False,
                )
            )
