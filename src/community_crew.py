"""Community Crew — OODA-Loop Orchestrierung (CrewAI-inspiriert).

Implementiert den vollständigen autonomen OODA-Zyklus:
  Observe  → Collector Agent  : Lädt und bereitet Social-Media-Daten vor
  Orient   → Analysis Agent   : Clustering + Community-Readiness-Bewertung
  Decide   → Action Agent     : Generiert ethisch gestaltete Vernetzungsvorschläge
  Act      → Ethics Agent     : Prüft Nudges, speichert nur APPROVED in final_nudges.json

Hinweis: CrewAI 0.11.2 ist nicht mit Python 3.14 kompatibel (Pydantic-v1-Konflikt).
Diese Datei implementiert dieselben Konzepte (Agent, Task, Crew, @tool, Process.sequential)
mit einer leichtgewichtigen, dependency-freien Orchestrierungsschicht.

Ausführen:
    venv/bin/python src/community_crew.py [--query "Fitness"]
"""

from __future__ import annotations

import json
import logging
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import yaml

# Scope warning filters — only suppress known noisy libraries
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", module="hdbscan")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pfad-Setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR  = PROJECT_ROOT / "results"
CONFIG_PATH  = PROJECT_ROOT / "configs" / "config.yaml"

sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_dotenv  # noqa: E402

load_dotenv()


# ---------------------------------------------------------------------------
# Lightweight CrewAI-kompatible Abstraktion
# ---------------------------------------------------------------------------

def tool(name: str) -> Callable:
    """Decorator: Markiert eine Funktion als CrewAI-Tool (kompatible API)."""
    def decorator(fn: Callable) -> Callable:
        fn._tool_name = name
        fn._is_tool = True
        return fn
    return decorator


@dataclass
class Agent:
    """CrewAI-kompatibler Agent mit Rolle, Ziel und Tools.

    Tool-Selection: Wenn nur ein Tool vorhanden ist, wird dieses direkt
    aufgerufen.  Bei mehreren Tools wird anhand des *tool_name*-Arguments
    das passende Tool gewählt (Dictionary-Lookup statt ``tools[0]``-Hardcoding).
    """
    role: str
    goal: str
    backstory: str
    tools: list[Callable] = field(default_factory=list)
    allow_delegation: bool = False
    verbose: bool = True

    def select_tool(self, tool_name: str | None = None) -> Callable:
        """Select a tool by name.  Falls back to the first tool."""
        if not self.tools:
            raise RuntimeError(f"[{self.role}] Keine Tools definiert.")
        if tool_name:
            by_name = {getattr(fn, "_tool_name", fn.__name__): fn for fn in self.tools}
            if tool_name in by_name:
                return by_name[tool_name]
        return self.tools[0]

    def run_tool(self, input_: str = "", tool_name: str | None = None) -> str:
        fn = self.select_tool(tool_name)
        if self.verbose:
            tname = getattr(fn, "_tool_name", fn.__name__)
            print(f"\n  [{self.role}] → {tname}")
        return fn(input_)


@dataclass
class Task:
    """CrewAI-kompatible Task-Definition.

    Context-Passing: Der Output vorheriger Tasks wird als String-Context
    aufgebaut und dem Tool übergeben.  Tools können den ``input_``-Parameter
    auswerten oder (wie aktuell) auf persistierte JSON-Dateien zurückgreifen —
    beides wird unterstützt.
    """
    description: str
    agent: Agent
    expected_output: str
    tool_name: str | None = None
    context: list["Task"] = field(default_factory=list)
    _output: str = field(default="", init=False)

    def context_summary(self) -> str:
        if not self.context:
            return ""
        parts = [f"Vorheriger Schritt — {t.agent.role}:\n{t._output}" for t in self.context]
        return "\n\n".join(parts)

    def execute(self, inputs: dict | None = None) -> str:
        # First task (no context): pass query from inputs; subsequent tasks: pass context
        if not self.context:
            arg = (inputs or {}).get("query", "")
        else:
            arg = self.context_summary()
        self._output = self.agent.run_tool(input_=arg, tool_name=self.tool_name)
        return self._output

    def reset(self) -> None:
        """Clear stored output from a previous run (prevents stale state)."""
        self._output = ""


class Process:
    sequential = "sequential"


@dataclass
class Crew:
    """CrewAI-kompatible Crew — führt Tasks sequentiell aus.

    Error-Recovery: Bricht die Pipeline bei einem Task-Fehler sauber ab
    und gibt den bisherigen Fortschritt zurück.
    """
    agents: list[Agent]
    tasks: list[Task]
    process: str = Process.sequential
    verbose: bool = True

    def kickoff(self, inputs: dict | None = None) -> str:
        _inputs = inputs or {}
        query_label = _inputs.get("query", "") or "—"
        n_tasks = len(self.tasks)
        print("\n" + "=" * 65)
        print(f" COMMUNITY CREW — OODA-Loop startet  (query='{query_label}')")
        print("=" * 65)

        # Reset tasks to prevent stale context from previous runs
        for task in self.tasks:
            task.reset()

        results = []
        for i, task in enumerate(self.tasks, 1):
            print(f"\n{'─' * 65}")
            print(f" Schritt {i}/{n_tasks} — {task.agent.role}")
            print(f" Aufgabe : {task.description[:80]}")
            print(f"{'─' * 65}")
            try:
                output = task.execute(inputs=_inputs)
            except Exception as exc:
                print(f"\n  ✗ FEHLER in Schritt {i}/{n_tasks} ({task.agent.role}): {exc}")
                results.append(f"=== {task.agent.role} === FEHLER\n{exc}")
                break
            results.append(f"=== {task.agent.role} ===\n{output}")
            if self.verbose:
                preview = output[:300] + ("..." if len(output) > 300 else "")
                print(f"\n  Ergebnis:\n  {preview}")

        print("\n" + "=" * 65)
        print(" CREW ABGESCHLOSSEN")
        print("=" * 65)
        return "\n\n".join(results)


# ---------------------------------------------------------------------------
# Tool 1: Collect Data (Observe) — via UniversalCollector
# ---------------------------------------------------------------------------

@tool("Collect Data")
def collect_data(query: str = "") -> str:
    """Lädt Posts via UniversalCollector (Reddit → Kaggle → Mock-Fallback).

    Args:
        query: Suchbegriff / Keyword. Leer = kein Filter (alle Posts).
    """
    from src.universal_collector import UniversalCollector

    collector = UniversalCollector()
    posts = collector.collect(query=query, limit=500)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "crew_posts.json").write_text(
        json.dumps(posts, indent=2, ensure_ascii=False)
    )

    cats = sorted({p["category"] for p in posts if p.get("category")})
    summary = {
        "n_posts": len(posts),
        "query":   query,
        "categories": cats,
        "data_path": str(RESULTS_DIR / "crew_posts.json"),
    }
    print(f"  ↳ {len(posts)} Posts gespeichert → {RESULTS_DIR / 'crew_posts.json'}")
    return json.dumps(summary, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tool 2: Run Semantic Analysis (Orient)
# ---------------------------------------------------------------------------

@tool("Run Semantic Analysis")
def run_analysis(input_: str = "") -> str:
    """Embeddings, Routing, Clustering und Readiness-Scores berechnen."""
    from src.embed import load_or_compute_embeddings, calculate_readiness_score
    from src.evaluate import truncate_mrl
    from src.routing import embed_label_prototypes, route_posts
    from src.hierarchical import cluster_per_topic

    cfg      = yaml.safe_load(CONFIG_PATH.read_text())
    model_cfg = cfg["model"]
    out_cfg   = cfg["output"]
    gemini_cfg = model_cfg.get("gemini", {}) or {}

    # Lade vorbereitete Posts
    records = pd.read_json(RESULTS_DIR / "crew_posts.json")

    # --- Embeddings (Gemini 3072, mit Cache) ---
    print("  ↳ Embeddings ...")
    embeddings, ids = load_or_compute_embeddings(
        records,
        text_col="text_clean",
        id_col="id",
        model_name=model_cfg["name"],
        batch_size=model_cfg.get("batch_size", 32),
        embeddings_path=str(PROJECT_ROOT / out_cfg["embeddings_path"]),
        ids_path=str(PROJECT_ROOT / out_cfg["ids_path"]),
        gemini_api_key=gemini_cfg.get("api_key"),
        gemini_output_dimensionality=gemini_cfg.get("output_dimensionality"),
        gemini_task_type=gemini_cfg.get("task_type", "CLUSTERING"),
    )

    # --- MRL-Truncation → 768 dims ---
    print("  ↳ MRL-Truncation → 768 dims ...")
    embeddings_768 = truncate_mrl(embeddings, 768)

    # --- Routing ---
    print("  ↳ Prototypen + Routing ...")
    interest_labels = cfg.get("interest_labels", {})
    hier_cfg = cfg.get("hierarchical", {})
    proto_cache = str(PROJECT_ROOT / out_cfg.get("proto_cache_dir", "results/proto_cache"))

    prototypes = embed_label_prototypes(
        interest_labels,
        model_name=model_cfg["name"],
        cache_dir=proto_cache,
        gemini_cfg={
            "api_key": gemini_cfg.get("api_key"),
            "output_dimensionality": gemini_cfg.get("output_dimensionality"),
            "task_type": gemini_cfg.get("task_type", "CLUSTERING"),
        } if model_cfg["name"].startswith("gemini") else None,
    )
    # Truncate prototypes to match 768-dim posts (MRL: first 768 dims + L2-norm)
    prototypes_768 = {}
    for label, proto in prototypes.items():
        p = proto[:768].copy()
        norm = np.linalg.norm(p)
        prototypes_768[label] = p / max(norm, 1e-12)

    routing_df = route_posts(embeddings_768, prototypes_768, top_n=2)
    routing_df.insert(0, "id", ids)

    # Merge mit Post-Texten
    df_merged = records.merge(routing_df, on="id", how="left")

    # --- Subclustering ---
    print("  ↳ Subclustering ...")
    topic_labels = df_merged["topic_1"].values
    hierarchy_df = cluster_per_topic(
        df_merged,
        embeddings_768,
        topic_labels,
        method=hier_cfg.get("subcluster_method", "hdbscan"),
        method_params=hier_cfg.get("subcluster_params", {}),
        kmeans_fallback_k=hier_cfg.get("kmeans_fallback_k", 4),
        min_samples_for_hdbscan=hier_cfg.get("min_samples_for_hdbscan", 10),
    )

    # --- Readiness-Scores ---
    print("  ↳ Readiness-Scores ...")
    readiness = calculate_readiness_score(
        embeddings_768,
        model_name=model_cfg["name"],
        gemini_output_dimensionality=768,
        gemini_task_type="SEMANTIC_SIMILARITY",
    )
    id_to_idx = {id_: i for i, id_ in enumerate(ids)}
    hierarchy_df = hierarchy_df.merge(
        df_merged[["id", "text_clean"]].rename(columns={}), on="id", how="left"
    )
    hierarchy_df["readiness_score"] = hierarchy_df["id"].map(
        lambda x: float(readiness[id_to_idx[x]]) if x in id_to_idx else 0.0
    )

    # --- hierarchy.csv persistieren (für Streamlit "Analysierte Posts"-Tab) ---
    hierarchy_df.to_csv(RESULTS_DIR / "hierarchy.csv", index=False)
    print(f"  ↳ Hierarchy gespeichert → {RESULTS_DIR / 'hierarchy.csv'}")

    # --- Cluster-Zusammenfassung bauen ---
    clusters = []
    for (topic, sub), grp in hierarchy_df.groupby(["topic_label", "subcluster_id"]):
        if int(sub) < 0:
            continue  # Noise
        top5 = grp.nlargest(5, "readiness_score")[
            ["id", "text_clean", "readiness_score"]
        ].to_dict("records")
        clusters.append({
            "topic_label": topic,
            "subcluster_id": int(sub),
            "n_posts": len(grp),
            "avg_readiness": round(float(grp["readiness_score"].mean()), 4),
            "top_posts": top5,
        })

    clusters.sort(key=lambda c: c["avg_readiness"], reverse=True)

    # --- Cluster-Plot generieren → results/cluster_plot.png ---
    print("  ↳ Cluster-Plot generieren ...")
    try:
        from src.visualize import reduce_dimensions, plot_topic_separation
        coords = reduce_dimensions(embeddings_768, method="tsne", random_state=42)
        plot_topic_separation(
            coords=coords,
            topic_labels=hierarchy_df["topic_label"].values,
            output_path=RESULTS_DIR / "cluster_plot.png",
            title="Community Cluster Analysis (Gemini 768 MRL · t-SNE)",
            dpi=150,
        )
    except Exception as exc:
        print(f"  ⚠ Plot-Fehler (nicht kritisch): {exc}")

    analysis_out = {
        "n_clusters": len(clusters),
        "n_posts_total": len(records),
        "mean_readiness": round(float(hierarchy_df["readiness_score"].mean()), 4),
        "clusters": clusters,
    }
    (RESULTS_DIR / "crew_analysis.json").write_text(
        json.dumps(analysis_out, indent=2, ensure_ascii=False)
    )
    print(f"  ↳ Analyse gespeichert → {RESULTS_DIR / 'crew_analysis.json'}")

    summary = {
        "n_clusters": len(clusters),
        "top_clusters": [
            {"topic": c["topic_label"], "sub": c["subcluster_id"],
             "avg_readiness": c["avg_readiness"], "n_posts": c["n_posts"]}
            for c in clusters[:5]
        ],
    }
    return json.dumps(summary, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tool 3: Generate Nudges (Decide)
# ---------------------------------------------------------------------------

@tool("Generate Community Nudges")
def generate_nudges(input_: str = "") -> str:
    """Generiert Vernetzungsvorschläge für Top-Cluster mit hoher Readiness."""
    from src.action_agent import ActionAgent

    analysis = json.loads((RESULTS_DIR / "crew_analysis.json").read_text())
    clusters = analysis["clusters"]

    # Top-3 Cluster mit avg_readiness > 0.35
    top_clusters = [c for c in clusters if c["avg_readiness"] > 0.35][:3]
    if not top_clusters:
        top_clusters = clusters[:3]

    agent = ActionAgent()
    nudges = []
    for cluster in top_clusters:
        print(f"  ↳ Nudge für {cluster['topic_label']}/sub{cluster['subcluster_id']} "
              f"(Ø Readiness={cluster['avg_readiness']}) ...")
        nudge = agent.generate_community_nudge(
            cluster_data={
                "topic_label": cluster["topic_label"],
                "subcluster_id": cluster["subcluster_id"],
                "n_posts": cluster["n_posts"],
            },
            top_posts=cluster["top_posts"],
        )
        nudges.append(nudge)

    (RESULTS_DIR / "crew_nudges.json").write_text(
        json.dumps(nudges, indent=2, ensure_ascii=False)
    )
    print(f"  ↳ {len(nudges)} Nudge(s) gespeichert → {RESULTS_DIR / 'crew_nudges.json'}")

    summary = {"n_nudges": len(nudges), "nudges": nudges}
    return json.dumps(summary, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tool 4: Validate Nudges (Act / Filter)
# ---------------------------------------------------------------------------

@tool("Validate Nudges Ethically")
def validate_nudges(input_: str = "") -> str:
    """Prüft alle Nudges auf Responsible-AI-Policies, speichert APPROVED in final_nudges.json."""
    from src.ethics_agent import EthicsAgent

    nudges = json.loads((RESULTS_DIR / "crew_nudges.json").read_text())
    ethics = EthicsAgent(output_dir=str(RESULTS_DIR / "ethics_reviews"))

    approved = []
    stats = {"total": len(nudges), "approved": 0, "revised": 0, "rejected": 0}

    for nudge in nudges:
        review = ethics.review_nudge(nudge)
        decision = review["decision"]

        if decision == "APPROVED":
            approved.append({**nudge, "ethics_decision": "APPROVED",
                              "ethics_reasoning": review["reasoning"]})
            stats["approved"] += 1
        elif decision == "REVISE" and review.get("modified_text"):
            revised_nudge = {**nudge, "nudge_text": review["modified_text"],
                             "ethics_decision": "REVISED",
                             "ethics_reasoning": review["reasoning"],
                             "original_nudge_text": nudge.get("nudge_text")}
            approved.append(revised_nudge)
            stats["revised"] += 1
        else:
            stats["rejected"] += 1
            print(f"  ↳ REJECTED: {review['reasoning'][:80]}...")

    final = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "total_reviewed": stats["total"],
        "approved_count": stats["approved"] + stats["revised"],
        "rejected_count": stats["rejected"],
        "nudges": approved,
    }
    (RESULTS_DIR / "final_nudges.json").write_text(
        json.dumps(final, indent=2, ensure_ascii=False)
    )
    print(f"  ↳ {len(approved)} Nudge(s) approved → {RESULTS_DIR / 'final_nudges.json'}")

    return json.dumps({
        "total_reviewed": stats["total"],
        "approved": stats["approved"],
        "revised": stats["revised"],
        "rejected": stats["rejected"],
        "output_path": str(RESULTS_DIR / "final_nudges.json"),
    }, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Agenten-Definitionen
# ---------------------------------------------------------------------------

collector = Agent(
    role="Data Collector (Observe)",
    goal="Lade und bereite Social-Media-Posts für die Analyse vor.",
    backstory=(
        "Du bist ein zuverlässiger Datensammler, der rohe Social-Media-Daten "
        "lädt, bereinigt und strukturiert für die nachgelagerte Analyse bereitstellt."
    ),
    tools=[collect_data],
)

analyst = Agent(
    role="Semantic Analyst (Orient)",
    goal="Identifiziere semantische Cluster und bewerte die Community-Readiness.",
    backstory=(
        "Du bist ein Experte für NLP und semantische Analyse. Du verwendest "
        "State-of-the-Art Embeddings (Gemini MRL) um Interessengruppen zu erkennen."
    ),
    tools=[run_analysis],
)

action_planner = Agent(
    role="Community Action Planner (Decide)",
    goal="Generiere transparente, ethisch gestaltete Vernetzungsvorschläge für Top-Cluster.",
    backstory=(
        "Du planst Community-Aktionen nach dem Value-Sensitive Design Prinzip. "
        "Deine Vorschläge sind diskret, transparent und autonomieerhaltend."
    ),
    tools=[generate_nudges],
)

ethics_reviewer = Agent(
    role="Ethics Reviewer (Act / Filter)",
    goal="Prüfe alle Nudges auf Responsible-AI-Policies und lass nur APPROVED passieren.",
    backstory=(
        "Du bist ein unabhängiger KI-Ethik-Prüfer. Du schützt Nutzer vor Manipulation "
        "und sorgst dafür, dass nur ethisch einwandfreie Vorschläge ausgespielt werden."
    ),
    tools=[validate_nudges],
)


# ---------------------------------------------------------------------------
# Task-Definitionen (Process.sequential)
# ---------------------------------------------------------------------------

task_collect = Task(
    description=(
        "Rufe das collect_data-Tool auf um die Social-Media-Posts zu laden "
        "und für die Analyse vorzubereiten."
    ),
    agent=collector,
    expected_output="JSON-Zusammenfassung der geladenen Posts (Anzahl, Kategorien).",
)

task_analyse = Task(
    description=(
        "Rufe das run_analysis-Tool auf um Embeddings, Topic-Routing, "
        "Subclustering und Readiness-Scores zu berechnen."
    ),
    agent=analyst,
    expected_output="JSON-Zusammenfassung der identifizierten Cluster mit Readiness-Werten.",
    context=[task_collect],
)

task_nudge = Task(
    description=(
        "Rufe das generate_nudges-Tool auf um für die Top-Cluster "
        "ethisch gestaltete Vernetzungsvorschläge zu generieren."
    ),
    agent=action_planner,
    expected_output="JSON mit generierten Nudges.",
    context=[task_analyse],
)

task_ethics = Task(
    description=(
        "Rufe das validate_nudges-Tool auf um alle Nudges ethisch zu prüfen "
        "und nur APPROVED-Ergebnisse in results/final_nudges.json zu speichern."
    ),
    agent=ethics_reviewer,
    expected_output="Abschlussbericht: Anzahl APPROVED, REVISED, REJECTED Nudges.",
    context=[task_nudge],
)


# ---------------------------------------------------------------------------
# Crew zusammenstellen
# ---------------------------------------------------------------------------

crew = Crew(
    agents=[collector, analyst, action_planner, ethics_reviewer],
    tasks=[task_collect, task_analyse, task_nudge, task_ethics],
    process=Process.sequential,
    verbose=True,
)


# ===========================================================================
# Agentic Coordinator — ersetzt die fixe Crew.kickoff()-Pipeline
# ===========================================================================

from src.agent_state import AgentState


def _collect_stateful(state: AgentState) -> AgentState:
    """Observe — Daten sammeln via UniversalCollector (AgentState-basiert)."""
    from src.universal_collector import UniversalCollector

    collector_inst = UniversalCollector()
    state.posts = collector_inst.collect(
        query=state.query,
        limit=state.limit,
        subreddits=state.subreddits,
    )

    state.add_message("Collector", "Coordinator", "DATA_READY", {
        "n_posts": len(state.posts),
        "query": state.query,
    })
    print(f"  ↳ {len(state.posts)} Posts gesammelt")
    return state


def _analyse_stateful(state: AgentState, output_dir: Path | None = None) -> AgentState:
    """Orient — Embeddings, Routing, Clustering, Readiness (AgentState-basiert)."""
    from src.embed import load_or_compute_embeddings, calculate_readiness_score
    from src.evaluate import truncate_mrl
    from src.routing import embed_label_prototypes, route_posts
    from src.hierarchical import cluster_per_topic

    cfg       = yaml.safe_load(CONFIG_PATH.read_text())
    model_cfg = cfg["model"]
    out_cfg   = cfg["output"]
    gemini_cfg = model_cfg.get("gemini", {}) or {}

    # Posts aus State statt von Disk lesen
    records = pd.DataFrame(state.posts)

    # --- Embeddings ---
    print("  ↳ Embeddings ...")
    embeddings, ids = load_or_compute_embeddings(
        records, text_col="text_clean", id_col="id",
        model_name=model_cfg["name"],
        batch_size=model_cfg.get("batch_size", 32),
        embeddings_path=str(PROJECT_ROOT / out_cfg["embeddings_path"]),
        ids_path=str(PROJECT_ROOT / out_cfg["ids_path"]),
        gemini_api_key=gemini_cfg.get("api_key"),
        gemini_output_dimensionality=gemini_cfg.get("output_dimensionality"),
        gemini_task_type=gemini_cfg.get("task_type", "CLUSTERING"),
    )

    # --- MRL-Truncation → 768 dims ---
    print("  ↳ MRL-Truncation → 768 dims ...")
    embeddings_768 = truncate_mrl(embeddings, 768)

    # --- Routing ---
    print("  ↳ Prototypen + Routing ...")
    interest_labels = cfg.get("interest_labels", {})
    hier_cfg = cfg.get("hierarchical", {})
    proto_cache = str(PROJECT_ROOT / out_cfg.get("proto_cache_dir", "results/proto_cache"))

    prototypes = embed_label_prototypes(
        interest_labels, model_name=model_cfg["name"], cache_dir=proto_cache,
        gemini_cfg={
            "api_key": gemini_cfg.get("api_key"),
            "output_dimensionality": gemini_cfg.get("output_dimensionality"),
            "task_type": gemini_cfg.get("task_type", "CLUSTERING"),
        } if model_cfg["name"].startswith("gemini") else None,
    )
    prototypes_768 = {}
    for label, proto in prototypes.items():
        p = proto[:768].copy()
        norm = np.linalg.norm(p)
        prototypes_768[label] = p / max(norm, 1e-12)

    routing_df = route_posts(embeddings_768, prototypes_768, top_n=2)
    routing_df.insert(0, "id", ids)
    df_merged = records.merge(routing_df, on="id", how="left")

    # --- Subclustering ---
    print("  ↳ Subclustering ...")
    topic_labels = df_merged["topic_1"].values
    hierarchy_df = cluster_per_topic(
        df_merged, embeddings_768, topic_labels,
        method=hier_cfg.get("subcluster_method", "hdbscan"),
        method_params=hier_cfg.get("subcluster_params", {}),
        kmeans_fallback_k=hier_cfg.get("kmeans_fallback_k", 4),
        min_samples_for_hdbscan=hier_cfg.get("min_samples_for_hdbscan", 10),
    )

    # --- Readiness-Scores ---
    print("  ↳ Readiness-Scores ...")
    readiness = calculate_readiness_score(
        embeddings_768, model_name=model_cfg["name"],
        gemini_output_dimensionality=768, gemini_task_type="SEMANTIC_SIMILARITY",
    )
    id_to_idx = {id_: i for i, id_ in enumerate(ids)}
    hierarchy_df = hierarchy_df.merge(
        df_merged[["id", "text_clean"]].rename(columns={}), on="id", how="left"
    )
    hierarchy_df["readiness_score"] = hierarchy_df["id"].map(
        lambda x: float(readiness[id_to_idx[x]]) if x in id_to_idx else 0.0
    )

    # Hierarchy in State speichern (für persist())
    state.hierarchy_df_records = hierarchy_df.to_dict("records")

    # --- Cluster-Zusammenfassung ---
    clusters = []
    for (topic, sub), grp in hierarchy_df.groupby(["topic_label", "subcluster_id"]):
        if int(sub) < 0:
            continue
        top5 = grp.nlargest(5, "readiness_score")[
            ["id", "text_clean", "readiness_score"]
        ].to_dict("records")
        clusters.append({
            "topic_label": topic, "subcluster_id": int(sub),
            "n_posts": len(grp),
            "avg_readiness": round(float(grp["readiness_score"].mean()), 4),
            "top_posts": top5,
        })
    clusters.sort(key=lambda c: c["avg_readiness"], reverse=True)

    # --- Cluster-Plot ---
    _plot_dir = output_dir or RESULTS_DIR
    print("  ↳ Cluster-Plot generieren ...")
    try:
        from src.visualize import reduce_dimensions, plot_topic_separation
        coords = reduce_dimensions(embeddings_768, method="tsne", random_state=42)
        _plot_dir.mkdir(parents=True, exist_ok=True)
        plot_topic_separation(
            coords=coords, topic_labels=hierarchy_df["topic_label"].values,
            output_path=_plot_dir / "cluster_plot.png",
            title="Community Cluster Analysis (Gemini 768 MRL · t-SNE)", dpi=150,
        )
    except Exception as exc:
        print(f"  ⚠ Plot-Fehler (nicht kritisch): {exc}")

    state.analysis = {
        "n_clusters": len(clusters), "n_posts_total": len(records),
        "mean_readiness": round(float(hierarchy_df["readiness_score"].mean()), 4),
        "clusters": clusters,
    }
    state.clusters = clusters

    state.add_message("Analyst", "Coordinator", "ANALYSIS_READY", {
        "n_clusters": len(clusters),
        "mean_readiness": state.analysis["mean_readiness"],
    })
    print(f"  ↳ {len(clusters)} Cluster identifiziert")
    return state


def _generate_nudges_stateful(state: AgentState) -> AgentState:
    """Decide — Nudges generieren aus State (AgentState-basiert)."""
    from src.action_agent import ActionAgent

    n = state.max_nudges
    top_clusters = [c for c in state.clusters if c["avg_readiness"] > 0.35][:n]
    if not top_clusters:
        top_clusters = state.clusters[:n]

    agent_inst = ActionAgent()
    nudges = []
    for cluster in top_clusters:
        topic_id = f"{cluster['topic_label']}/sub{cluster['subcluster_id']}"
        print(f"  ↳ Nudge für {topic_id} "
              f"(Ø Readiness={cluster['avg_readiness']}) ...")
        try:
            nudge = agent_inst.generate_community_nudge(
                cluster_data={
                    "topic_label": cluster["topic_label"],
                    "subcluster_id": cluster["subcluster_id"],
                    "n_posts": cluster["n_posts"],
                },
                top_posts=cluster["top_posts"],
            )
        except Exception as exc:
            print(f"  ✗ ActionAgent-Fehler für {topic_id}: {exc}")
            state.errors.append({
                "phase": "nudge_generation",
                "topic": topic_id,
                "error": str(exc)[:200],
            })
            continue
        # Speichere Top-Posts im Nudge für den Revision-Loop (Coordinator)
        nudge["_top_posts"] = cluster["top_posts"]
        nudges.append(nudge)

    state.nudges = nudges
    state.add_message("ActionPlanner", "Coordinator", "NUDGES_READY", {
        "n_nudges": len(nudges),
    })
    print(f"  ↳ {len(nudges)} Nudge(s) generiert")
    return state


# ---------------------------------------------------------------------------
# Coordinator — Agentic Orchestrator mit Feedback-Loop
# ---------------------------------------------------------------------------

class Coordinator:
    """Regelbasierter Orchestrator der den OODA-Loop mit Inter-Agent-
    Kommunikation und Feedback-Loops steuert.

    Im Gegensatz zu ``Crew.kickoff()`` (fixe sequentielle Pipeline):
    - Agents kommunizieren über Shared State statt File-I/O
    - Ethics-Agent kann abgelehnte Nudges an Action-Agent zurücksenden
    - Iterative Verbesserung: Action → Ethics → Action → ... (max N)
    - Message-Log für Audit-Trail
    - Flexible Phase-Callbacks für UI-Integration
    """

    def __init__(self, max_revisions: int = 3, verbose: bool = True) -> None:
        self.max_revisions = max_revisions
        self.verbose = verbose

    def run(
        self,
        state: AgentState,
        on_phase: Optional[Callable] = None,
        output_dir: Path | None = None,
    ) -> AgentState:
        """Führe den vollständigen OODA-Loop mit Feedback aus.

        Args:
            state: AgentState mit mindestens ``query`` gesetzt.
            on_phase: Optionaler Callback ``(phase: str, detail: str) -> None``
                      für UI-Fortschrittsanzeige (z.B. Streamlit).
            output_dir: Zielverzeichnis für Ergebnisse. Default: RESULTS_DIR.

        Returns:
            AgentState mit allen Ergebnissen.
        """
        _out = Path(output_dir) if output_dir else RESULTS_DIR

        def _notify(phase: str, detail: str = "") -> None:
            state.current_phase = phase
            if on_phase:
                on_phase(phase, detail)

        print("\n" + "=" * 65)
        print(f" COORDINATOR — OODA-Loop startet  (query='{state.query or '—'}')")
        print("=" * 65)

        # --- Phase 1: Observe ---
        _notify("collect", "Daten sammeln...")
        print(f"\n{'─' * 65}\n Observe — Collector\n{'─' * 65}")
        try:
            state = _collect_stateful(state)
        except Exception as exc:
            state.errors.append({"phase": "collect", "error": str(exc)})
            state.add_message("Coordinator", "log", "ERROR", {"phase": "collect", "error": str(exc)})
            print(f"  ✗ Collector-Fehler: {exc}")
            return state

        if not state.posts:
            state.add_message("Coordinator", "log", "WARNING", {"msg": "Keine Posts gesammelt"})
            print("  ⚠ Keine Posts — Abbruch.")
            return state

        # --- Phase 2: Orient ---
        _notify("analyse", "Clustering & Readiness-Scores...")
        print(f"\n{'─' * 65}\n Orient — Analyst\n{'─' * 65}")
        try:
            state = _analyse_stateful(state, output_dir=_out)
        except Exception as exc:
            state.errors.append({"phase": "analyse", "error": str(exc)})
            state.add_message("Coordinator", "log", "ERROR", {"phase": "analyse", "error": str(exc)})
            print(f"  ✗ Analyse-Fehler: {exc}")
            return state

        if not state.clusters:
            state.add_message("Coordinator", "log", "WARNING", {"msg": "Keine Cluster gefunden"})
            print("  ⚠ Keine Cluster — Abbruch.")
            return state

        # --- Phase 3: Decide ---
        _notify("nudge", "Nudges generieren...")
        print(f"\n{'─' * 65}\n Decide — Action Planner\n{'─' * 65}")
        try:
            state = _generate_nudges_stateful(state)
        except Exception as exc:
            state.errors.append({"phase": "nudge", "error": str(exc)})
            state.add_message("Coordinator", "log", "ERROR", {"phase": "nudge", "error": str(exc)})
            print(f"  ✗ Nudge-Fehler: {exc}")
            return state

        # --- Phase 4: Act + Feedback-Loop ---
        _notify("ethics", "Ethische Validierung + Feedback-Loop...")
        print(f"\n{'─' * 65}\n Act — Ethics Reviewer (mit Feedback-Loop)\n{'─' * 65}")
        try:
            state = self._ethics_with_revision_loop(state, output_dir=_out)
        except Exception as exc:
            state.errors.append({"phase": "ethics", "error": str(exc)})
            state.add_message("Coordinator", "log", "ERROR", {"phase": "ethics", "error": str(exc)})
            print(f"  ✗ Ethics-Fehler: {exc}")

        # --- Persistieren ---
        _notify("persist", "Ergebnisse speichern...")
        _out.mkdir(parents=True, exist_ok=True)
        state.persist(_out)

        state.current_phase = "done"
        n_final = len(state.final_nudges)
        n_total = len(state.nudges)
        print(f"\n{'=' * 65}")
        print(f" COORDINATOR ABGESCHLOSSEN — {n_final}/{n_total} Nudges freigegeben")
        print(f" Messages: {len(state.messages)} | Errors: {len(state.errors)}")
        print(f"{'=' * 65}")

        return state

    def _ethics_with_revision_loop(self, state: AgentState, output_dir: Path | None = None) -> AgentState:
        """Prüfe jeden Nudge; abgelehnte werden an Action-Agent zurückgesendet."""
        from src.ethics_agent import EthicsAgent

        _out = output_dir or RESULTS_DIR
        ethics = EthicsAgent(output_dir=str(_out / "ethics_reviews"))

        for nudge in state.nudges:
            if not nudge.get("nudge_text"):
                state.add_message("EthicsAgent", "Coordinator", "SKIPPED", {
                    "reason": "Leerer nudge_text",
                })
                continue

            try:
                review = ethics.review_nudge(nudge)
            except Exception as exc:
                topic = nudge.get("cluster_context", {}).get("topic_label", "?")
                print(f"  ✗ Ethics-Fehler für {topic}: {exc}")
                state.errors.append({
                    "phase": "ethics",
                    "topic": topic,
                    "error": str(exc)[:200],
                })
                state.add_message("EthicsAgent", "Coordinator", "ERROR", {
                    "topic": topic,
                    "error": str(exc)[:100],
                })
                continue

            decision = review["decision"]

            if decision == "APPROVED":
                state.final_nudges.append({
                    **nudge,
                    "ethics_decision": "APPROVED",
                    "ethics_reasoning": review["reasoning"],
                    "revision_count": 0,
                })
                state.add_message("EthicsAgent", "Coordinator", "APPROVED", {
                    "topic": nudge.get("cluster_context", {}).get("topic_label", "?"),
                })
                print(f"  ✅ APPROVED (1. Durchlauf)")

            elif decision == "REVISE" and review.get("modified_text"):
                state.final_nudges.append({
                    **nudge,
                    "nudge_text": review["modified_text"],
                    "ethics_decision": "REVISED",
                    "ethics_reasoning": review["reasoning"],
                    "original_nudge_text": nudge.get("nudge_text"),
                    "revision_count": 0,
                })
                state.add_message("EthicsAgent", "Coordinator", "REVISED", {
                    "topic": nudge.get("cluster_context", {}).get("topic_label", "?"),
                })
                print(f"  🔄 REVISED (Ethics hat Text angepasst)")

            else:
                # REJECTED → Feedback-Loop starten
                print(f"  ❌ REJECTED — starte Revision-Loop...")
                state = self._revision_loop(state, nudge, review, ethics)

        return state

    def _revision_loop(
        self,
        state: AgentState,
        nudge: dict,
        initial_review: dict,
        ethics: Any,
    ) -> AgentState:
        """Iterativer Revisions-Zyklus: Action → Ethics → Action → ...

        Args:
            state: Aktueller AgentState.
            nudge: Der abgelehnte Nudge.
            initial_review: Das erste Ethics-Review (REJECTED).
            ethics: EthicsAgent-Instanz.

        Returns:
            Aktualisierter AgentState.
        """
        from src.action_agent import ActionAgent

        action_agent = ActionAgent()
        current_nudge = nudge
        current_review = initial_review

        for attempt in range(1, self.max_revisions + 1):
            state.iteration = attempt

            # Ethics → Action: Feedback senden
            state.add_message("EthicsAgent", "ActionAgent", "REVISE_REQUEST", {
                "feedback": current_review["reasoning"],
                "attempt": attempt,
                "max_attempts": self.max_revisions,
                "topic": nudge.get("cluster_context", {}).get("topic_label", "?"),
            })

            print(f"    Revision {attempt}/{self.max_revisions} — "
                  f"Action-Agent regeneriert mit Feedback...")

            # Action-Agent: Neuen Nudge generieren mit Feedback
            revised_nudge = action_agent.regenerate_nudge(
                cluster_data=nudge.get("cluster_context", {}),
                top_posts=nudge.get("_top_posts", []),
                previous_nudge_text=current_nudge.get("nudge_text", ""),
                ethics_feedback=current_review["reasoning"],
            )

            # Action → Ethics: Revidierten Nudge senden
            state.add_message("ActionAgent", "EthicsAgent", "REVISED_NUDGE", {
                "attempt": attempt,
                "new_text": revised_nudge.get("nudge_text", "")[:80],
            })

            # Ethics: Neuen Nudge prüfen
            new_review = ethics.review_nudge(revised_nudge)
            new_decision = new_review["decision"]

            if new_decision == "APPROVED":
                state.final_nudges.append({
                    **revised_nudge,
                    "ethics_decision": "APPROVED",
                    "ethics_reasoning": new_review["reasoning"],
                    "revision_count": attempt,
                })
                state.add_message("EthicsAgent", "Coordinator", "APPROVED_AFTER_REVISION", {
                    "attempts": attempt,
                })
                print(f"    ✅ APPROVED nach {attempt} Revision(en)")
                return state

            elif new_decision == "REVISE" and new_review.get("modified_text"):
                state.final_nudges.append({
                    **revised_nudge,
                    "nudge_text": new_review["modified_text"],
                    "ethics_decision": "REVISED",
                    "ethics_reasoning": new_review["reasoning"],
                    "original_nudge_text": revised_nudge.get("nudge_text"),
                    "revision_count": attempt,
                })
                state.add_message("EthicsAgent", "Coordinator", "REVISED_AFTER_REVISION", {
                    "attempts": attempt,
                })
                print(f"    🔄 REVISED nach {attempt} Revision(en)")
                return state

            # Weiterhin REJECTED → nächste Iteration
            current_nudge = revised_nudge
            current_review = new_review
            print(f"    ❌ Weiterhin REJECTED (Versuch {attempt})")

        # Max Iterationen erreicht
        state.add_message("Coordinator", "log", "MAX_REVISIONS_REACHED", {
            "topic": nudge.get("cluster_context", {}).get("topic_label", "?"),
            "attempts": self.max_revisions,
            "last_feedback": current_review.get("reasoning", "")[:100],
        })
        print(f"    ⚠ Max Revisionen ({self.max_revisions}) erreicht — Nudge verworfen")
        return state


# ---------------------------------------------------------------------------
# Coordinator-Instanz
# ---------------------------------------------------------------------------

coordinator = Coordinator(max_revisions=3, verbose=True)


def run_ooda_loop(
    query: str = "",
    on_phase: Optional[Callable] = None,
    output_dir: Path | None = None,
    limit: int = 500,
    subreddits: list[str] | None = None,
    max_nudges: int = 3,
) -> AgentState:
    """Convenience-Funktion: Erstellt AgentState und startet den Coordinator."""
    state = AgentState(
        query=query,
        limit=limit,
        subreddits=subreddits,
        max_nudges=max_nudges,
    )
    return coordinator.run(state, on_phase=on_phase, output_dir=output_dir)


# ---------------------------------------------------------------------------
# Haupteinsprungpunkt
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Community OODA-Loop (Agentic Coordinator)")
    parser.add_argument(
        "--query", default="",
        help="Suchbegriff für Datenbeschaffung (z.B. 'Fitness', 'health'). "
             "Leer = alle Posts laden.",
    )
    parser.add_argument(
        "--limit", type=int, default=500,
        help="Maximale Anzahl Posts zum Laden (default: 500).",
    )
    parser.add_argument(
        "--subreddits", default=None,
        help="Komma-separierte Subreddit-Liste (z.B. 'technology,art,books'). "
             "Default: alle verfügbaren.",
    )
    parser.add_argument(
        "--max-nudges", type=int, default=3,
        help="Maximale Anzahl Nudges zum Generieren (default: 3).",
    )
    parser.add_argument(
        "--max-revisions", type=int, default=3,
        help="Maximale Revisions-Iterationen pro abgelehntem Nudge.",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Zielverzeichnis für Ergebnisse (z.B. 'results/neu'). "
             "Default: results/",
    )
    parser.add_argument(
        "--legacy", action="store_true",
        help="Legacy-Modus: Verwendet die alte Crew.kickoff()-Pipeline.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else None
    subs = [s.strip() for s in args.subreddits.split(",")] if args.subreddits else None

    if args.legacy:
        # --- Legacy: Crew.kickoff() (Wasserfall) ---
        result = crew.kickoff(inputs={"query": args.query})
    else:
        # --- Agentic: Coordinator mit Feedback-Loop ---
        coord = Coordinator(max_revisions=args.max_revisions)
        state = AgentState(
            query=args.query,
            limit=args.limit,
            subreddits=subs,
            max_nudges=args.max_nudges,
        )
        state = coord.run(state, output_dir=out_dir)

        # Ergebnis anzeigen
        _result_path = out_dir or RESULTS_DIR
        snapshot = state.snapshot()
        print(f"\n Finale Ausgabe: {_result_path / 'final_nudges.json'}")
        print(f" Posts: {snapshot['n_posts']} | Cluster: {snapshot['n_clusters']} | "
              f"Nudges: {snapshot['n_final_nudges']}/{snapshot['n_nudges']}")
        print(f" Messages: {snapshot['n_messages']} | Errors: {snapshot['n_errors']}")

        for n in state.final_nudges:
            ctx = n.get("cluster_context", {})
            rev = n.get("revision_count", 0)
            rev_info = f" (nach {rev} Revision{'en' if rev != 1 else ''})" if rev > 0 else ""
            print(f"\n [{ctx.get('topic_label','?')} / sub{ctx.get('subcluster_id','?')}] "
                  f"{n['ethics_decision']}{rev_info}")
            print(f"  {n['nudge_text']}")
