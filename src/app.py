"""Social AI Dashboard — OODA-Loop Forschungs-Dashboard (ZHAW Bachelorarbeit 2026).

Startet mit:
    .venv/bin/streamlit run src/app.py
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
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------------------------------------------------------------------------
# Header + Status-Streifen
# ---------------------------------------------------------------------------

st.title("🧠 Social AI Dashboard")
st.caption("ZHAW Bachelorarbeit · Multi-Agent Community Detection · OODA-Loop · Ethics-by-Design")

# Status-Streifen: dynamisch aus manifest.json
_corpus_path = PROJECT_ROOT / "data" / "corpus_index"
_manifest = None
if (_corpus_path / "manifest.json").exists():
    try:
        _manifest = json.loads((_corpus_path / "manifest.json").read_text())
    except Exception:
        pass

col_s1, col_s2, col_s3 = st.columns(3)
if _manifest:
    col_s1.metric("Korpus", f"{_manifest.get('n_posts', '?'):,} Posts · Evolution-AI".replace(",", "'"))
    col_s2.metric("Embedding", f"Gemini {_manifest.get('dim', '?')}d MRL")
    col_s3.metric("Index-Status", "✅ geladen")
else:
    col_s1.metric("Korpus", "—")
    col_s2.metric("Embedding", "—")
    col_s3.metric("Index-Status", "❌ fehlt", help="Bitte zuerst: python scripts/build_corpus_index.py")

st.divider()


# ---------------------------------------------------------------------------
# Shared: Phase-Icons für Pipeline-Callbacks
# ---------------------------------------------------------------------------

_PHASE_INFO = {
    "collect":  ("🔍", "Collector", "Observe", "Daten sammeln..."),
    "analyse":  ("📊", "Analyst", "Orient", "Clustering & Readiness-Scores..."),
    "nudge":    ("💬", "Action Planner", "Decide", "Nudges generieren..."),
    "ethics":   ("⚖️", "Ethics Reviewer", "Act", "Ethische Validierung + Feedback-Loop..."),
    "persist":  ("💾", "Coordinator", "Persist", "Ergebnisse speichern..."),
}

_PHASE_INFO_QUERY = {
    **_PHASE_INFO,
    "collect": ("🔍", "Discovery", "Retrieve", "Posts aus Korpus laden..."),
}

_MSG_ICONS = {
    "APPROVED": "✅", "REVISED": "🔄", "REVISE_REQUEST": "❌→🔁",
    "REJECTED": "❌", "ERROR": "⚠️", "DATA_READY": "📦",
    "ANALYSIS_READY": "📊", "NUDGES_READY": "💬", "SKIPPED": "⏭️",
}


# ---------------------------------------------------------------------------
# Shared: Ergebnis-Anzeige (Nudges + Analytics)
# ---------------------------------------------------------------------------

def _load_ethics_scores() -> dict:
    """Load ethics review scores, keyed by nudge_text (first 80 chars)."""
    scores_map = {}
    reviews_dir = RESULTS_DIR / "ethics_reviews"
    if reviews_dir.exists():
        for f in reviews_dir.glob("review_*.json"):
            try:
                r = json.loads(f.read_text())
                key = (r.get("input_nudge", {}).get("nudge_text") or "")[:80]
                if key and r.get("scores"):
                    scores_map[key] = {
                        "scores": r["scores"],
                        "avg_score": r.get("avg_score"),
                        "decision": r.get("decision"),
                    }
            except Exception:
                pass
    return scores_map


def _render_results():
    """Rendert die Ergebnis-Tabs basierend auf den gespeicherten JSON/CSV-Dateien."""

    tab_nudges, tab_analytics = st.tabs(["🚀 Community Nudges", "📊 Technical Analytics"])

    # === Shared data loading ===
    _posts_by_id: dict = {}
    _hierarchy_df = None
    _readiness_by_id: dict = {}
    _posts_path = RESULTS_DIR / "crew_posts.json"
    _hier_path = RESULTS_DIR / "hierarchy.csv"
    _analysis_path = RESULTS_DIR / "crew_analysis.json"
    if _posts_path.exists():
        try:
            _all_posts = json.loads(_posts_path.read_text())
            _posts_by_id = {p["id"]: p for p in _all_posts}
        except Exception:
            _all_posts = []
    else:
        _all_posts = []
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
            _analysis = None
    else:
        _analysis = None

    ethics_scores = _load_ethics_scores()

    # ===================================================================
    # Tab 1: Community Nudges
    # ===================================================================
    with tab_nudges:
        final_path = RESULTS_DIR / "final_nudges.json"
        if not final_path.exists():
            st.info("Noch keine Ergebnisse. Bitte eine Analyse starten.")
        else:
            try:
                final = json.loads(final_path.read_text())
            except Exception:
                st.error("final_nudges.json konnte nicht gelesen werden.")
                final = None

            if final:
                m1, m2, m3 = st.columns(3)
                m1.metric("Geprüfte Nudges", final.get("total_reviewed", 0))
                m2.metric("Freigegeben", final.get("approved_count", 0),
                          help="APPROVED + REVISED Nudges")
                m3.metric("Abgelehnt", final.get("rejected_count", 0),
                          help="Nudges die gegen Responsible-AI-Policies verstossen")
                st.divider()

                nudges = final.get("nudges", [])
                if not nudges:
                    st.warning("Keine freigegebenen Nudges in diesem Lauf.")
                else:
                    for i, nudge in enumerate(nudges):
                        ctx = nudge.get("cluster_context") or {}
                        decision = nudge.get("ethics_decision", "—")
                        topic = ctx.get("topic_label", "?")
                        sub = ctx.get("subcluster_id", "?")
                        eligible_ids = {str(x) for x in nudge.get("target_user_ids", [])}

                        if _hierarchy_df is not None:
                            _mask = ((_hierarchy_df["topic_label"] == topic) &
                                     (_hierarchy_df["subcluster_id"] == sub))
                            n_posts = int(_mask.sum())
                        else:
                            n_posts = ctx.get("n_posts", "?")

                        # Ethics score lookup (try current text, then original text for REVISED nudges)
                        nudge_key = (nudge.get("nudge_text") or "")[:80]
                        eth = ethics_scores.get(nudge_key, {})
                        if not eth and nudge.get("original_nudge_text"):
                            nudge_key_orig = (nudge["original_nudge_text"])[:80]
                            eth = ethics_scores.get(nudge_key_orig, {})
                        avg_score = eth.get("avg_score")
                        score_text = f" · Score: {avg_score:.1f}/10" if avg_score else ""

                        with st.container(border=True):
                            # Header: Nudge text + meta
                            col_text, col_meta = st.columns([3, 1])
                            with col_text:
                                st.markdown(f"**{nudge.get('nudge_text', '—')}**")
                            with col_meta:
                                badge_color = "green" if decision == "APPROVED" else "orange"
                                rev_count = nudge.get("revision_count", 0)
                                rev_text = f"  \n**Revisionen:** {rev_count}" if rev_count > 0 else ""
                                st.markdown(
                                    f"**Thema:** {topic} / sub{sub}  \n"
                                    f"**Cluster:** {n_posts} Posts  \n"
                                    f"*(davon {len(eligible_ids)} bereit)*  \n"
                                    f"**Ethik:** :{badge_color}[{decision}]{score_text}"
                                    f"{rev_text}"
                                )

                            # Sub-tabs: Begründung | Ethik-Scores | Posts
                            sub_tabs = ["📖 Begründung", "📊 Ethik-Scores", f"📝 Posts ({n_posts})"]
                            tab_reason, tab_ethics, tab_posts = st.tabs(sub_tabs)

                            with tab_reason:
                                st.markdown(f"**Erklärung:** {nudge.get('explanation', '—')}")
                                st.markdown(f"**Ethics Reasoning:** {nudge.get('ethics_reasoning', '—')}")

                            with tab_ethics:
                                scores = eth.get("scores", {})
                                if scores:
                                    # 7 Kriterien als Metriken in 2 Zeilen
                                    criteria_de = {
                                        "manipulation": "Manipulations-vermeidung",
                                        "transparenz": "Transparenz",
                                        "diskriminierung": "Diskriminierungs-vermeidung",
                                        "autonomie": "Autonomie",
                                        "relevanz": "Relevanz",
                                        "spezifitaet": "Spezifität",
                                        "mehrwert": "Mehrwert",
                                    }
                                    cols = st.columns(4)
                                    for j, (key, label) in enumerate(criteria_de.items()):
                                        val = scores.get(key, "—")
                                        cols[j % 4].metric(label, f"{val}/10")
                                    if avg_score:
                                        st.caption(f"Durchschnitt: {avg_score:.2f}/10 → {eth.get('decision', '—')}")
                                else:
                                    st.info("Keine detaillierten Ethik-Scores verfügbar.")

                            with tab_posts:
                                if _hierarchy_df is not None and _posts_by_id:
                                    mask = ((_hierarchy_df["topic_label"] == topic) &
                                            (_hierarchy_df["subcluster_id"] == sub))
                                    cluster_ids = _hierarchy_df.loc[mask, "id"].tolist()
                                    cluster_posts = [_posts_by_id[pid] for pid in cluster_ids
                                                     if pid in _posts_by_id]
                                    cluster_posts.sort(
                                        key=lambda p: (p["id"] not in eligible_ids,
                                                       -_readiness_by_id.get(p["id"], 0))
                                    )
                                    if cluster_posts:
                                        rows = []
                                        for p in cluster_posts:
                                            pid = p["id"]
                                            is_elig = pid in eligible_ids
                                            readiness = _readiness_by_id.get(pid, p.get("readiness_score"))
                                            row = {
                                                "Status": "🟢" if is_elig else "",
                                                "Text": p["text_clean"][:150] + ("..." if len(p["text_clean"]) > 150 else ""),
                                                "Readiness": round(readiness, 3) if readiness else None,
                                                "Kategorie": p.get("category", "—"),
                                            }
                                            qsim = p.get("query_similarity")
                                            if qsim is not None:
                                                row["Query-Sim"] = round(qsim, 3)
                                            rows.append(row)
                                        df_posts = pd.DataFrame(rows)
                                        st.dataframe(
                                            df_posts,
                                            use_container_width=True,
                                            height=400,
                                            column_config={
                                                "Readiness": st.column_config.ProgressColumn(
                                                    "Readiness", min_value=0.0, max_value=1.0, format="%.3f"),
                                                "Query-Sim": st.column_config.ProgressColumn(
                                                    "Query-Sim", min_value=0.0, max_value=1.0, format="%.3f")
                                                if "Query-Sim" in df_posts.columns else None,
                                            },
                                        )
                                    else:
                                        st.info("Keine Posts für diesen Cluster gefunden.")
                                else:
                                    st.info("Analyse-Daten nicht vorhanden — bitte Analyse starten.")

    # ===================================================================
    # Tab 2: Technical Analytics
    # ===================================================================
    with tab_analytics:
        if _analysis is None:
            st.info("Noch keine Analyse-Daten. Bitte eine Analyse starten.")
        else:
            # --- Sektion A: Metriken ---
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Analysierte Posts", _analysis.get("n_posts_total", "—"))
            m2.metric("Gefundene Cluster", _analysis.get("n_clusters", "—"))
            m3.metric("Ø Community-Readiness", f"{_analysis.get('mean_readiness', 0):.3f}",
                      help="Ø Kosinusähnlichkeit zur Anker-Phrase (Gemini 768 MRL)")
            # Clustering-Methode aus hierarchy
            if _hierarchy_df is not None and "method_used" in _hierarchy_df.columns:
                methods = _hierarchy_df["method_used"].value_counts()
                primary = methods.index[0] if len(methods) > 0 else "—"
                m4.metric("Clustering", primary.upper().replace("_", " "))
            else:
                m4.metric("Clustering", "—")

            st.divider()

            # --- Sektion B: Visualisierungen ---
            st.subheader("Cluster-Visualisierung")
            plot_path = RESULTS_DIR / "cluster_plot.png"
            heatmap_path = RESULTS_DIR / "readiness_heatmap.png"

            if heatmap_path.exists() and plot_path.exists():
                col_tsne, col_heat = st.columns(2)
                with col_tsne:
                    st.image(str(plot_path), use_container_width=True,
                             caption="t-SNE-Projektion eingefärbt nach Topic-Routing-Zuordnung. "
                                     "Jede Farbe entspricht einer thematischen Kategorie.")
                with col_heat:
                    st.image(str(heatmap_path), use_container_width=True,
                             caption="Community-Readiness-Heatmap: Gelb = hohe Vernetzungsbereitschaft, "
                                     "Dunkel = niedrig. Basierend auf Kosinus-Ähnlichkeit zum Anker-Satz.")
            elif plot_path.exists():
                col_plot_single, _ = st.columns([2, 1])
                with col_plot_single:
                    st.image(str(plot_path), use_container_width=True,
                             caption="t-SNE-Projektion der Post-Embeddings (Gemini 768 MRL), "
                                     "eingefärbt nach Topic-Label. Jeder Punkt ist ein Post.")
            else:
                st.info("Cluster-Plot nicht vorhanden.")

            st.divider()

            # --- Sektion C: Verteilungen ---
            st.subheader("Verteilungen")
            if _hierarchy_df is not None:
                col_cat, col_read = st.columns(2)

                with col_cat:
                    st.markdown("**Kategorie-Verteilung**")
                    cat_counts = _hierarchy_df["topic_label"].value_counts().sort_values(ascending=True)
                    st.bar_chart(cat_counts)

                with col_read:
                    st.markdown("**Readiness-Score-Verteilung**")
                    if "readiness_score" in _hierarchy_df.columns:
                        import numpy as np
                        scores = _hierarchy_df["readiness_score"].dropna()
                        hist, bin_edges = np.histogram(scores, bins=20)
                        bin_labels = [f"{bin_edges[i]:.2f}" for i in range(len(hist))]
                        hist_df = pd.DataFrame({"Score-Bereich": bin_labels, "Anzahl": hist})
                        st.bar_chart(hist_df.set_index("Score-Bereich"))
                    else:
                        st.info("Readiness-Scores nicht verfügbar.")
            else:
                st.info("Keine Hierarchy-Daten für Verteilungsanalyse.")

            st.divider()

            # --- Sektion D: Cluster-Details ---
            st.subheader("Cluster-Details")
            clusters = _analysis.get("clusters", [])
            if clusters and _hierarchy_df is not None:
                detail_rows = []
                for c in clusters:
                    topic = c["topic_label"]
                    sub_id = c["subcluster_id"]
                    mask = ((_hierarchy_df["topic_label"] == topic) &
                            (_hierarchy_df["subcluster_id"] == sub_id))
                    sub_df = _hierarchy_df[mask]
                    method = sub_df["method_used"].iloc[0] if len(sub_df) > 0 and "method_used" in sub_df.columns else "—"
                    # Count noise for this topic
                    topic_mask = _hierarchy_df["topic_label"] == topic
                    noise = int((_hierarchy_df[topic_mask]["subcluster_id"] == -1).sum()) if sub_id >= 0 else 0
                    detail_rows.append({
                        "Thema": topic,
                        "Subcluster": f"sub{sub_id}",
                        "Posts": c["n_posts"],
                        "Noise": noise,
                        "Methode": method.replace("_", " ").title(),
                        "Ø Readiness": c["avg_readiness"],
                    })
                df_detail = pd.DataFrame(detail_rows).sort_values("Ø Readiness", ascending=False).reset_index(drop=True)
                st.dataframe(df_detail, use_container_width=True,
                             column_config={"Ø Readiness": st.column_config.ProgressColumn(
                                 "Ø Readiness", min_value=0.0, max_value=1.0, format="%.3f")})
            elif clusters:
                df_simple = pd.DataFrame([
                    {"Thema": c["topic_label"], "Subcluster": f"sub{c['subcluster_id']}",
                     "Posts": c["n_posts"], "Ø Readiness": c["avg_readiness"]}
                    for c in clusters
                ]).sort_values("Ø Readiness", ascending=False).reset_index(drop=True)
                st.dataframe(df_simple, use_container_width=True)
            else:
                st.info("Keine Cluster-Daten verfügbar.")

            # --- Sektion E: Query-Relevanz (nur wenn query_similarity vorhanden) ---
            if _all_posts and _all_posts[0].get("query_similarity") is not None:
                st.divider()
                st.subheader("Query-Relevanz (Top-10)")
                sorted_by_sim = sorted(_all_posts, key=lambda p: p.get("query_similarity", 0), reverse=True)[:10]
                sim_rows = [{
                    "Text": p["text_clean"][:120] + "...",
                    "Similarity": round(p["query_similarity"], 4),
                    "Kategorie": p.get("category", "—"),
                } for p in sorted_by_sim]
                st.dataframe(pd.DataFrame(sim_rows), use_container_width=True,
                             column_config={"Similarity": st.column_config.ProgressColumn(
                                 "Similarity", min_value=0.0, max_value=1.0, format="%.4f")})


# ---------------------------------------------------------------------------
# Shared: Agent-Kommunikation anzeigen
# ---------------------------------------------------------------------------

def _render_agent_messages(state):
    """Zeigt den Audit Trail der Inter-Agenten-Kommunikation."""
    if state and state.messages:
        with st.expander(f"📨 Agent-Kommunikation ({len(state.messages)} Messages)", expanded=False):
            for msg in state.messages:
                ts = msg.timestamp if hasattr(msg, "timestamp") else "—"
                icon = _MSG_ICONS.get(msg.message_type, "📩")
                st.markdown(
                    f"`{ts}` {icon} **{msg.sender}** → **{msg.receiver}**: "
                    f"`{msg.message_type}` {json.dumps(msg.payload, ensure_ascii=False)[:120]}"
                )


# ---------------------------------------------------------------------------
# Shared: Pipeline-Status anzeigen
# ---------------------------------------------------------------------------

def _render_pipeline_status(state):
    """Zeigt Zusammenfassung nach Pipeline-Durchlauf."""
    if state.errors:
        for err in state.errors:
            st.error(f"Fehler in Phase '{err['phase']}': {err['error']}")
        return "⚠️ Abgeschlossen mit Fehlern", "error"
    else:
        n_final = len(state.final_nudges)
        n_total = len(state.nudges)
        n_revisions = sum(n.get("revision_count", 0) for n in state.final_nudges)
        label = f"✅ {n_final}/{n_total} Nudges freigegeben"
        if n_revisions > 0:
            label += f" ({n_revisions} Revision{'en' if n_revisions != 1 else ''})"
        return label, "complete"


# ---------------------------------------------------------------------------
# Mode Tabs
# ---------------------------------------------------------------------------

tab_query, tab_batch = st.tabs(["🔍 Themen-Suche (Query)", "📊 Korpus-Analyse (Batch)"])


# ── Query Mode ──────────────────────────────────────────────────────────────

with tab_query:
    query_q = st.text_input(
        "Suchanfrage",
        placeholder="z.B. Immobilien-Investments, Marathon-Training, Burnout-Prävention ...",
        help=(
            "Semantische Suche im Evolution-AI-Korpus. "
            "Die thematisch passendsten Beiträge werden durch die OODA-Pipeline analysiert."
        ),
        key="query_input",
    )
    with st.expander("Erweiterte Optionen", expanded=False):
        col_topk, col_sim = st.columns(2)
        with col_topk:
            top_k = st.slider("Top-K Ergebnisse", 50, 500, 200, step=50,
                               help="Anzahl der semantisch ähnlichsten Posts.")
        with col_sim:
            min_sim = st.slider("Min. Ähnlichkeit", 0.0, 1.0, 0.0, step=0.05,
                                 help="0.0 = deaktiviert. Höhere Werte filtern strenger.")

    run_query = st.button("🔍 Communities entdecken", type="primary",
                          use_container_width=True, key="btn_query")

    state_q = None
    if run_query and query_q.strip():
        from src.discovery import discover

        def _on_phase_q(phase: str, detail: str = "") -> None:
            info = _PHASE_INFO_QUERY.get(phase)
            if info:
                icon, agent, step, desc = info
                st.write(f"{icon} **{agent}** ({step}): {detail or desc}")

        with st.status(f"🔍 Suche: '{query_q}' ...", expanded=True) as status_q:
            try:
                state_q = discover(query_q, top_k=top_k, min_similarity=min_sim,
                                   on_phase=_on_phase_q)
            except FileNotFoundError as e:
                status_q.update(label="❌ Corpus-Index fehlt", state="error")
                st.error(str(e))
                state_q = None
            except Exception as e:
                status_q.update(label="⚠️ Fehler", state="error")
                st.error(f"Discovery-Fehler: {e}")
                state_q = None

            if state_q is not None:
                label, s = _render_pipeline_status(state_q)
                n_posts = len(state_q.posts) if state_q.posts else 0
                status_q.update(label=f"{label} · {n_posts} Posts analysiert", state=s)

        # Kategorie-Verteilung
        if state_q and state_q.posts:
            _cats = pd.Series([p.get("category", "?") for p in state_q.posts]).value_counts()
            _cat_str = ", ".join(f"**{k}** ({v})" for k, v in _cats.head(5).items())
            st.info(f"🏷 Top-Kategorien der Treffer: {_cat_str}")

        _render_agent_messages(state_q)

    elif run_query and not query_q.strip():
        st.warning("Bitte eine Suchanfrage eingeben.")

    # Ergebnisse anzeigen (aus gespeicherten Dateien)
    _render_results()


# ── Batch Mode ──────────────────────────────────────────────────────────────

with tab_batch:
    query_b = st.text_input(
        "Keyword / Thema",
        placeholder="z.B. health, Fitness, AI, Kunst ...",
        help=(
            "Filtert Posts nach diesem Suchbegriff. "
            "Leer lassen = alle verfügbaren Posts laden."
        ),
        key="batch_input",
    )
    run_batch = st.button("🚀 Analyse starten", type="primary",
                          use_container_width=True, key="btn_batch")

    state_b = None
    if run_batch:
        from src.agent_state import AgentState
        from src.community_crew import coordinator

        state_b = AgentState(query=query_b)

        def _on_phase_b(phase: str, detail: str = "") -> None:
            info = _PHASE_INFO.get(phase)
            if info:
                icon, agent, step, desc = info
                st.write(f"{icon} **{agent}** ({step}): {detail or desc}")

        with st.status("OODA-Loop läuft...", expanded=True) as status_b:
            state_b = coordinator.run(state_b, on_phase=_on_phase_b)
            label, s = _render_pipeline_status(state_b)
            status_b.update(label=label, state=s)

        _render_agent_messages(state_b)

    # Ergebnisse anzeigen
    _render_results()


# ---------------------------------------------------------------------------
# Info-Expander am Seitenende
# ---------------------------------------------------------------------------

st.divider()
with st.expander("ℹ️ Über das System — Methodik, Datenschutz & Architektur"):

    st.subheader("🔬 Methodik & Value-Sensitive Design")
    st.markdown(
        "Das System generiert Vernetzungsvorschläge nach drei Kernprinzipien:\n\n"
        "- **Unaufdringlich** — keine FOMO-Sprache, kein Druck\n"
        "- **Transparent** — jeder Vorschlag enthält eine ehrliche Begründung\n"
        "- **Autonom** — Nutzende entscheiden frei, ob sie die Verbindung möchten"
    )

    st.subheader("🔒 Datenschutz")
    st.markdown(
        "Benutzernamen werden **niemals im Klartext gespeichert**. "
        "Stattdessen wird ein SHA-256-Hash mit festem Salt (`zhaw_2026`) erzeugt. "
        "Dies ermöglicht Konsistenzprüfungen ohne Rückschluss auf den Klarnamen."
    )

    st.subheader("🤖 Agenten-Architektur (OODA-Loop)")
    st.markdown(
        "```\n"
        "Observe  → Collector Agent    (Daten sammeln)\n"
        "Orient   → Analysis Agent     (Clustering & Readiness)\n"
        "Decide   → Action Agent       (Nudges generieren)\n"
        "Act      → Ethics Agent       (7-Kriterien-Prüfung)\n"
        "         ↺ Feedback-Loop      (max. 3 Revisionen)\n"
        "```"
    )
    st.caption("Agentic Coordinator · Shared State · Gemini 768 MRL · Ethics-by-Design")
