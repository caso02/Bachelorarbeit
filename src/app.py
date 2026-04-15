"""Social AI Dashboard — OODA-Loop Forschungs-Dashboard (ZHAW Bachelorarbeit 2026).

Startet mit:
    venv/bin/streamlit run src/app.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Pfad-Setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR  = PROJECT_ROOT / "results"
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_dotenv  # noqa: E402

load_dotenv()


# ---------------------------------------------------------------------------
# Seiten-Konfiguration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Social AI Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🔬 Methodik")
    st.divider()

    # Collector-Status
    st.subheader("Datenquelle")
    if os.environ.get("REDDIT_CLIENT_ID"):
        st.success("🟢 Reddit API aktiv")
    elif (PROJECT_ROOT / "data" / "kaggle" / "reddit_data.csv").exists():
        st.warning("🟡 Kaggle-Daten aktiv")
    else:
        st.error("🔴 Mock-Fallback (300 synthetische Posts)")

    st.divider()

    # VSD-Erklärung
    st.subheader("Value-Sensitive Design (VSD)")
    st.markdown(
        "Das System generiert Vernetzungsvorschläge nach drei Kernprinzipien:\n\n"
        "- **Unaufdringlich** — keine FOMO-Sprache, kein Druck\n"
        "- **Transparent** — jeder Vorschlag enthält eine ehrliche Begründung\n"
        "- **Autonom** — Nutzer entscheidet frei, ob er/sie die Verbindung möchte"
    )
    st.divider()

    # Datenschutz
    st.subheader("🔒 Datenschutz")
    st.markdown(
        "Usernamen werden **niemals im Klartext gespeichert**. "
        "Stattdessen wird ein SHA-256-Hash mit festem Salt (`zhaw_2026`) erzeugt:\n\n"
        "```\nSHA-256('zhaw_2026' + username)[:16]\n```\n\n"
        "Dies ermöglicht Konsistenzprüfungen ohne Rückschluss auf den Klarnamen."
    )
    st.divider()

    # System-Info
    st.subheader("🤖 Agenten-Architektur")
    st.markdown(
        "```\nObserve  → Collector Agent\n"
        "Orient   → Analysis Agent\n"
        "Decide   → Action Agent\n"
        "Act      → Ethics Agent\n"
        "         ↺ Feedback-Loop\n```"
    )
    st.caption("Agentic Coordinator · Shared State · Feedback-Loop · Gemini 768 MRL")


# ---------------------------------------------------------------------------
# Hauptbereich — Header
# ---------------------------------------------------------------------------

st.title("🧠 Social AI Dashboard")
st.markdown(
    "Autonomes Multi-Agent-System zur semantischen Analyse von Social-Media-Posts "
    "und ethisch gestalteter Community-Vernetzung."
)
st.divider()

# ---------------------------------------------------------------------------
# Input & Execution
# ---------------------------------------------------------------------------

col_input, col_btn = st.columns([4, 1])
with col_input:
    query = st.text_input(
        "Keyword / Thema",
        placeholder="z.B. health, Fitness, AI, Kunst ...",
        help=(
            "Filtert Posts nach diesem Suchbegriff. "
            "Leer lassen = alle verfügbaren Posts laden.\n\n"
            "Bei aktiver Reddit-API wird das Keyword direkt für die Suche verwendet."
        ),
    )
with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("🚀 Analyse starten", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# OODA-Loop ausführen
# ---------------------------------------------------------------------------

if run_btn:
    # Lazy imports — erst bei Klick laden
    from src.agent_state import AgentState
    from src.community_crew import coordinator

    state = AgentState(query=query)

    # Phase-Icons für UI-Feedback
    _phase_info = {
        "collect":  ("🔍", "Collector", "Observe", "Daten sammeln..."),
        "analyse":  ("📊", "Analyst", "Orient", "Clustering & Readiness-Scores..."),
        "nudge":    ("💬", "Action Planner", "Decide", "Nudges generieren..."),
        "ethics":   ("⚖️", "Ethics Reviewer", "Act", "Ethische Validierung + Feedback-Loop..."),
        "persist":  ("💾", "Coordinator", "Persist", "Ergebnisse speichern..."),
    }

    with st.status("OODA-Loop läuft...", expanded=True) as status:
        _status_container = st.empty()

        def _on_phase(phase: str, detail: str = "") -> None:
            info = _phase_info.get(phase)
            if info:
                icon, agent, step, desc = info
                st.write(f"{icon} **{agent}** ({step}): {detail or desc}")

        state = coordinator.run(state, on_phase=_on_phase)

        # Ergebnis-Zusammenfassung
        if state.errors:
            status.update(label="⚠️ Abgeschlossen mit Fehlern", state="error")
            for err in state.errors:
                st.error(f"Fehler in Phase '{err['phase']}': {err['error']}")
        else:
            n_final = len(state.final_nudges)
            n_total = len(state.nudges)
            n_revisions = sum(n.get("revision_count", 0) for n in state.final_nudges)
            label = f"✅ Analyse abgeschlossen — {n_final}/{n_total} Nudges freigegeben"
            if n_revisions > 0:
                label += f" ({n_revisions} Revision{'en' if n_revisions != 1 else ''})"
            status.update(label=label, state="complete")

    # Agent-Kommunikation als Expander anzeigen
    if state.messages:
        with st.expander(f"📨 Agent-Kommunikation ({len(state.messages)} Messages)", expanded=False):
            for msg in state.messages:
                ts = msg.timestamp if hasattr(msg, "timestamp") else "—"
                icon = {"APPROVED": "✅", "REVISED": "🔄", "REVISE_REQUEST": "❌→🔁",
                         "REJECTED": "❌", "ERROR": "⚠️", "DATA_READY": "📦",
                         "ANALYSIS_READY": "📊", "NUDGES_READY": "💬"}.get(msg.message_type, "📩")
                st.markdown(
                    f"`{ts}` {icon} **{msg.sender}** → **{msg.receiver}**: "
                    f"`{msg.message_type}` {json.dumps(msg.payload, ensure_ascii=False)[:120]}"
                )

    st.rerun()

# ---------------------------------------------------------------------------
# Ergebnis-Tabs
# ---------------------------------------------------------------------------

tab_nudges, tab_analytics = st.tabs(["🚀 Community Nudges", "📊 Technical Analytics"])


# --- Tab 1: Community Nudges ------------------------------------------------

with tab_nudges:
    final_path = RESULTS_DIR / "final_nudges.json"

    if not final_path.exists():
        st.info("Noch keine Ergebnisse. Bitte oben eine Analyse starten.")
    else:
        try:
            final = json.loads(final_path.read_text())
        except Exception:
            st.error("final_nudges.json konnte nicht gelesen werden.")
            final = None

        # Posts + Cluster-Zuordnung laden (für "Analysierte Posts"-Tab)
        _posts_by_id: dict = {}
        _hierarchy_df: pd.DataFrame | None = None
        _readiness_by_id: dict = {}   # post_id (str) → readiness_score (float)
        _posts_path   = RESULTS_DIR / "crew_posts.json"
        _hier_path    = RESULTS_DIR / "hierarchy.csv"
        _analysis_path = RESULTS_DIR / "crew_analysis.json"
        if _posts_path.exists():
            try:
                _posts_by_id = {p["id"]: p for p in json.loads(_posts_path.read_text())}
            except Exception:
                pass
        if _hier_path.exists():
            try:
                _hierarchy_df = pd.read_csv(_hier_path, dtype={"id": str})
            except Exception:
                pass
        if _analysis_path.exists():
            try:
                _analysis = json.loads(_analysis_path.read_text())
                for _c in _analysis.get("clusters", []):
                    for _tp in _c.get("top_posts", []):
                        _readiness_by_id[str(_tp["id"])] = _tp["readiness_score"]
            except Exception:
                pass

        if final:
            # Zusammenfassung
            m1, m2, m3 = st.columns(3)
            m1.metric("Geprüfte Nudges", final.get("total_reviewed", 0))
            m2.metric("Freigegeben", final.get("approved_count", 0),
                      help="APPROVED + REVISED Nudges")
            m3.metric("Abgelehnt", final.get("rejected_count", 0),
                      help="Nudges die gegen Responsible-AI-Policies verstoßen")

            st.divider()

            nudges = final.get("nudges", [])
            if not nudges:
                st.warning("Keine freigegebenen Nudges in diesem Lauf.")
            else:
                for nudge in nudges:
                    ctx         = nudge.get("cluster_context") or {}
                    decision    = nudge.get("ethics_decision", "—")
                    topic       = ctx.get("topic_label", "?")
                    sub         = ctx.get("subcluster_id", "?")
                    eligible_ids = {str(i) for i in nudge.get("target_user_ids", [])}

                    # Tatsächliche Post-Anzahl aus Hierarchy ermitteln
                    if _hierarchy_df is not None:
                        _mask = (
                            (_hierarchy_df["topic_label"] == topic) &
                            (_hierarchy_df["subcluster_id"] == sub)
                        )
                        n_posts = int(_mask.sum())
                    else:
                        n_posts = ctx.get("n_posts", "?")

                    n_eligible = len(eligible_ids)

                    with st.container(border=True):
                        col_text, col_meta = st.columns([3, 1])

                        with col_text:
                            st.markdown(f"**{nudge.get('nudge_text', '—')}**")

                        with col_meta:
                            badge_color = "green" if decision == "APPROVED" else "orange"
                            rev_count = nudge.get("revision_count", 0)
                            rev_text = f"  \n**Revisionen:** {rev_count}" if rev_count > 0 else ""
                            st.markdown(
                                f"**Thema:** {topic} / sub{sub}  \n"
                                f"**Cluster-Stärke:** {n_posts} Posts gesamt  \n"
                                f"*(davon {n_eligible} vernetzungsbereit)*  \n"
                                f"**Ethik:** :{badge_color}[{decision}]"
                                f"{rev_text}"
                            )

                        tab_detail, tab_posts = st.tabs(["📖 Begründung & Ethik", "📝 Analysierte Posts"])

                        with tab_detail:
                            st.markdown(f"**Erklärung:** {nudge.get('explanation', '—')}")
                            st.markdown(f"**Ethics Reasoning:** {nudge.get('ethics_reasoning', '—')}")
                            if eligible_ids:
                                st.caption(f"Ziel-IDs (pseudonymisiert): {sorted(eligible_ids)}")

                        with tab_posts:
                            if _hierarchy_df is not None and _posts_by_id:
                                mask = (
                                    (_hierarchy_df["topic_label"] == topic) &
                                    (_hierarchy_df["subcluster_id"] == sub)
                                )
                                cluster_ids = _hierarchy_df.loc[mask, "id"].tolist()
                                cluster_posts = [
                                    _posts_by_id[pid]
                                    for pid in cluster_ids
                                    if pid in _posts_by_id
                                ]
                                # Vernetzungsbereite Posts zuerst
                                cluster_posts.sort(
                                    key=lambda p: (p["id"] not in eligible_ids,
                                                   -_readiness_by_id.get(p["id"], 0))
                                )
                                if cluster_posts:
                                    for p in cluster_posts:
                                        pid = p["id"]
                                        is_eligible = pid in eligible_ids
                                        readiness = _readiness_by_id.get(pid)

                                        if is_eligible:
                                            st.markdown(
                                                f"🟢 **{p['text_clean']}**"
                                            )
                                        else:
                                            st.markdown(f"> {p['text_clean']}")

                                        caption_parts = [f"Kategorie: {p.get('category', '—')}"]
                                        if readiness is not None:
                                            caption_parts.append(f"Readiness: {readiness:.3f}")
                                        if is_eligible:
                                            caption_parts.append("✅ vernetzungsbereit")
                                        st.caption(" · ".join(caption_parts))
                                        st.divider()
                                else:
                                    st.info("Keine Posts für diesen Cluster gefunden.")
                            else:
                                st.info("crew_posts.json oder hierarchy.csv nicht vorhanden — Analyse starten.")


# --- Tab 2: Technical Analytics ---------------------------------------------

with tab_analytics:
    analysis_path = RESULTS_DIR / "crew_analysis.json"

    if not analysis_path.exists():
        st.info("Noch keine Analyse-Daten. Bitte oben eine Analyse starten.")
    else:
        try:
            analysis = json.loads(analysis_path.read_text())
        except Exception:
            st.error("crew_analysis.json konnte nicht gelesen werden.")
            analysis = None

        if analysis:
            # Metriken
            m1, m2, m3 = st.columns(3)
            m1.metric(
                "Analysierte Posts",
                analysis.get("n_posts_total", "—"),
                help="Anzahl Posts nach Preprocessing und Keyword-Filter",
            )
            m2.metric(
                "Gefundene Cluster",
                analysis.get("n_clusters", "—"),
                help=(
                    "Hierarchische Untercluster über alle Themen. "
                    "Ermittelt via HDBSCAN (mit KMeans-Fallback bei kleinen Gruppen)."
                ),
            )
            m3.metric(
                "Ø Community-Readiness",
                f"{analysis.get('mean_readiness', 0):.3f}",
                help=(
                    "Durchschnittliche Kosinusähnlichkeit zur Anker-Phrase "
                    "'Ich möchte mich mit Gleichgesinnten vernetzen'. "
                    "Berechnet via Gemini 768 MRL-Embedding "
                    "(Matryoshka Representation Learning — erste 768 Dims des 3072-Dim-Vektors)."
                ),
            )

            st.divider()

            col_plot, col_table = st.columns([3, 2])

            # Cluster-Plot
            with col_plot:
                st.subheader("t-SNE Projektion")
                plot_path = RESULTS_DIR / "cluster_plot.png"
                if plot_path.exists():
                    st.image(
                        str(plot_path),
                        caption=(
                            "t-SNE Projektion der Post-Embeddings (Gemini 768 MRL). "
                            "Farben entsprechen den Stage-1 Topic-Labels."
                        ),
                        use_container_width=True,
                    )
                else:
                    st.info("Cluster-Plot noch nicht vorhanden — Analyse starten.")

            # Cluster-Tabelle
            with col_table:
                st.subheader("Cluster-Übersicht")
                clusters = analysis.get("clusters", [])
                if clusters:
                    df_clusters = pd.DataFrame([
                        {
                            "Thema":      c["topic_label"],
                            "Subcluster": f"sub{c['subcluster_id']}",
                            "Posts":      c["n_posts"],
                            "Ø Readiness": c["avg_readiness"],
                        }
                        for c in clusters
                    ]).sort_values("Ø Readiness", ascending=False).reset_index(drop=True)

                    st.dataframe(
                        df_clusters,
                        use_container_width=True,
                        column_config={
                            "Ø Readiness": st.column_config.ProgressColumn(
                                "Ø Readiness",
                                help="Ø Kosinusähnlichkeit zur Community-Anker-Phrase",
                                min_value=0.0,
                                max_value=1.0,
                                format="%.3f",
                            )
                        },
                    )
                else:
                    st.info("Keine Cluster-Daten verfügbar.")
