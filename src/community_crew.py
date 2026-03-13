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
import os
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Pfad-Setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR  = PROJECT_ROOT / "results"
CONFIG_PATH  = PROJECT_ROOT / "configs" / "config.yaml"

sys.path.insert(0, str(PROJECT_ROOT))

# .env laden
_env_file = PROJECT_ROOT / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())


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
    """CrewAI-kompatibler Agent mit Rolle, Ziel und Tools."""
    role: str
    goal: str
    backstory: str
    tools: list[Callable] = field(default_factory=list)
    allow_delegation: bool = False
    verbose: bool = True

    def run_tool(self, fn: Callable, input_: str = "") -> str:
        if self.verbose:
            tname = getattr(fn, "_tool_name", fn.__name__)
            print(f"\n  [{self.role}] → {tname}")
        return fn(input_)


@dataclass
class Task:
    """CrewAI-kompatible Task-Definition."""
    description: str
    agent: Agent
    expected_output: str
    context: list["Task"] = field(default_factory=list)
    _output: str = field(default="", init=False)

    def context_summary(self) -> str:
        if not self.context:
            return ""
        parts = [f"Vorheriger Schritt — {t.agent.role}:\n{t._output}" for t in self.context]
        return "\n\n".join(parts)

    def execute(self, inputs: dict | None = None) -> str:
        if not self.agent.tools:
            return f"[{self.agent.role}] Keine Tools definiert."
        fn = self.agent.tools[0]
        # First task (no context): pass query from inputs; subsequent tasks: pass context
        if not self.context:
            arg = (inputs or {}).get("query", "")
        else:
            arg = self.context_summary()
        self._output = self.agent.run_tool(fn, arg)
        return self._output


class Process:
    sequential = "sequential"


@dataclass
class Crew:
    """CrewAI-kompatible Crew — führt Tasks sequentiell aus."""
    agents: list[Agent]
    tasks: list[Task]
    process: str = Process.sequential
    verbose: bool = True

    def kickoff(self, inputs: dict | None = None) -> str:
        _inputs = inputs or {}
        query_label = _inputs.get("query", "") or "—"
        print("\n" + "=" * 65)
        print(f" COMMUNITY CREW — OODA-Loop startet  (query='{query_label}')")
        print("=" * 65)

        results = []
        for i, task in enumerate(self.tasks, 1):
            print(f"\n{'─' * 65}")
            print(f" Schritt {i}/4 — {task.agent.role}")
            print(f" Aufgabe : {task.description[:80]}")
            print(f"{'─' * 65}")
            output = task.execute(inputs=_inputs)
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

    analysis_out = {
        "n_clusters": len(clusters),
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


# ---------------------------------------------------------------------------
# Haupteinsprungpunkt
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Community OODA-Loop Crew")
    parser.add_argument(
        "--query", default="",
        help="Suchbegriff für Datenbeschaffung (z.B. 'Fitness', 'health'). "
             "Leer = alle Posts laden.",
    )
    args = parser.parse_args()

    result = crew.kickoff(inputs={"query": args.query})
    print("\n" + "=" * 65)
    print(" Finale Ausgabe: results/final_nudges.json")
    print("=" * 65)
    final_path = RESULTS_DIR / "final_nudges.json"
    if final_path.exists():
        final = json.loads(final_path.read_text())
        print(f" Geprüft: {final['total_reviewed']}  |  "
              f"Freigegeben: {final['approved_count']}  |  "
              f"Abgelehnt: {final['rejected_count']}")
        for n in final["nudges"]:
            ctx = n.get("cluster_context", {})
            print(f"\n [{ctx.get('topic_label','?')} / sub{ctx.get('subcluster_id','?')}] "
                  f"{n['ethics_decision']}")
            print(f"  {n['nudge_text']}")
