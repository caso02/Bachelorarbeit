# Social AI — Community OODA-Loop (ZHAW Bachelorarbeit 2026)

Autonomes Multi-Agent-System zur semantischen Analyse von Social-Media-Posts und ethisch gestalteter Community-Vernetzung.

## Architektur

Das System implementiert einen **OODA-Loop** (Observe-Orient-Decide-Act) mit fünf spezialisierten Agenten:

| Agent | OODA-Phase | Aufgabe |
|-------|-----------|---------|
| Collector | Observe | Datenerhebung aus Social Media (Reddit API / Kaggle / Mock) |
| Analyst | Orient | Gemini Embeddings, Topic-Routing, HDBSCAN-Subclustering, Readiness-Scores |
| Action | Decide | Ethisch gestaltete Vernetzungsvorschläge (Nudges) generieren |
| Ethics | Act / Filter | 7-Kriterien-Scoring, Revision oder Ablehnung |
| Coordinator | Orchestrierung | OODA-Loop-Steuerung, Feedback-Loop zwischen Action und Ethics |

Der **Coordinator-Agent** orchestriert den iterativen Feedback-Loop: Abgelehnte Nudges werden an den Action-Agent zurückgesendet (max. 3 Revisionen). Alle Agenten kommunizieren über ein geteiltes Zustandsobjekt (`AgentState`) mit strukturiertem Nachrichtensystem und Audit Trail.

### Value-Sensitive Design (VSD)

- **Unaufdringlich** — keine FOMO-Sprache, kein Druck
- **Transparent** — jeder Vorschlag enthält eine ehrliche Begründung
- **Autonomieerhaltend** — Nutzer entscheidet frei
- **Privatsphäre** — Usernamen werden SHA-256-pseudonymisiert, nie im Klartext gespeichert

## Zwei Modi

### OODA-Loop Modus
Keyword-basierte Analyse: Collector sammelt Posts, Pipeline läuft vollständig durch.

### Query-Modus (Semantische Suche)
Freitextliche Suchanfrage gegen einen vorberechneten Korpus-Index (97'500 Posts). Das Discovery-Modul selektiert die relevantesten Posts per Kosinus-Ähnlichkeit und führt sie durch die OODA-Pipeline.

## Projektstruktur

```
├── configs/
│   └── config.yaml              # Zentrale Konfiguration
├── data/
│   ├── raw/sample_posts_300.csv  # Mock-Dataset (401 Posts, 8 Kategorien)
│   ├── kaggle/                   # Optional: Kaggle Reddit CSV
│   └── corpus_index/             # Vorberechneter Korpus-Index (Query-Modus)
├── results/                      # Pipeline-Outputs (JSON, CSV, Plots)
├── scripts/
│   ├── run_pipeline.py           # Klassische Analyse-Pipeline (CLI)
│   ├── run_evaluation.py         # Evaluations-Suite (4 Läufe)
│   └── build_corpus_index.py     # Korpus-Index für Query-Modus erstellen
├── src/
│   ├── app.py                    # Streamlit Dashboard (OODA + Query-Modus)
│   ├── community_crew.py         # Coordinator + OODA-Loop Orchestrierung
│   ├── agent_state.py            # Geteiltes Zustandsobjekt + Nachrichtensystem
│   ├── universal_collector.py    # Datensammlung (Reddit → Kaggle → Mock)
│   ├── embed.py                  # Gemini / SBERT Embeddings (Batch, Cache)
│   ├── routing.py                # Topic-Routing via Prototyp-Embeddings
│   ├── hierarchical.py           # HDBSCAN / KMeans Subclustering
│   ├── action_agent.py           # Nudge-Generierung (Gemini LLM + VSD)
│   ├── ethics_agent.py           # 7-Kriterien Responsible-AI-Prüfung
│   ├── discovery.py              # Semantische Suche (Query-Modus)
│   ├── evaluate.py               # Metriken, TF-IDF, MRL-Truncation
│   ├── stability.py              # Stabilitätsanalyse (Bootstrap + Perturbation)
│   ├── model_comparison.py       # Embedding-Modellvergleich
│   ├── cluster.py                # KMeans, DBSCAN, HDBSCAN
│   ├── visualize.py              # t-SNE Plots, Readiness-Heatmaps
│   ├── preprocess.py             # Textbereinigung
│   ├── report.py                 # JSON/CSV/Markdown Reports
│   └── utils.py                  # Shared Utilities (.env, JSON-Parsing, Client-Cache)
├── tests/                        # 99 automatisierte Tests (pytest)
├── requirements.txt
├── .env.example                  # Template für Umgebungsvariablen
└── README.md
```

## Setup

**Voraussetzung:** Python 3.11+

```bash
# 1. Virtual Environment erstellen
python3 -m venv venv
source venv/bin/activate

# 2. Dependencies installieren
pip install -r requirements.txt

# 3. Umgebungsvariablen konfigurieren
cp .env.example .env
# .env editieren: GOOGLE_API_KEY eintragen (Pflicht)
# Optional: REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET für Live-Daten
```

## Quickstart

### Streamlit Dashboard (empfohlen)

```bash
streamlit run src/app.py
```

Öffnet ein interaktives Dashboard mit zwei Modi:

**OODA-Loop Modus:**
- Keyword-Suche für Topic-Filterung
- Schrittweiser OODA-Loop mit Live-Status
- Community Nudges mit Ethics-Badge (APPROVED / REVISED / REJECTED)
- Ethik-Scores: 7 Kriterien (Manipulation, Transparenz, Diskriminierung, Autonomie, Relevanz, Spezifität, Mehrwert)

**Query-Modus:**
- Freitextliche semantische Suche (z.B. "mental health support")
- Selektiert Posts aus 97'500 vorindexierten Beiträgen
- Durchläuft anschliessend die vollständige OODA-Pipeline

**Beide Modi zeigen:**
- Community Nudges mit Begründung und Ethik-Bewertung
- Technical Analytics: t-SNE Cluster-Plot, Readiness-Heatmap, Cluster-Tabelle

### CLI (Coordinator mit Feedback-Loop)

```bash
python src/community_crew.py --query "health" --max-nudges 3
```

### Korpus-Index erstellen (für Query-Modus)

```bash
python scripts/build_corpus_index.py
# Smoketest: python scripts/build_corpus_index.py --n-per-category 10
```

### Klassische Analyse-Pipeline

```bash
python scripts/run_pipeline.py --config configs/config.yaml
```

## Tests

```bash
python -m pytest tests/ -v
# 99 Tests in 7 Testdateien
```

## Datenquellen (Prioritäts-Fallback)

| Modus | Bedingung | Beschreibung |
|-------|-----------|--------------|
| Reddit API | `REDDIT_CLIENT_ID` in `.env` | Live-Suche via praw |
| Kaggle CSV | `data/kaggle/*.csv` vorhanden | Statische Reddit-Daten |
| Mock-Fallback | Immer verfügbar | 401 synthetische Posts (8 Kategorien) |

## Pipeline-Outputs

| Datei | Beschreibung |
|-------|--------------|
| `results/final_nudges.json` | Freigegebene Nudges (nur APPROVED/REVISED) |
| `results/hierarchy.csv` | Post-Cluster-Zuordnungen |
| `results/crew_analysis.json` | Cluster mit Readiness-Scores |
| `results/ethics_reviews/` | Einzelne Ethics-Review-Protokolle |
| `results/cluster_plot.png` | t-SNE Visualisierung |
| `results/readiness_heatmap.png` | Readiness-Heatmap |
| `results/agent_messages.json` | Audit Trail der Inter-Agent-Kommunikation |

## Konfiguration

Alle Parameter werden über `configs/config.yaml` gesteuert:

- **model.name** — Embedding-Modell (`gemini-embedding-2-preview` oder SBERT)
- **interest_labels** — 8 Topic-Kategorien mit je 10-12 Seed-Sätzen
- **hierarchical** — HDBSCAN/KMeans Parameter für Subclustering
- **query_mode** — Korpus-Index-Pfad, Top-K, Mindestähnlichkeit
- **output** — Pfade für Cache und Ergebnisse

## Reproduzierbarkeit

- Embeddings werden gecacht (`results/embeddings.npy` + `results/embedding_ids.json`)
- Label-Prototypen gecacht (`results/proto_cache/`)
- Korpus-Index mit Manifest (Modell, Dimensionalität, Hash, Timestamp)
- Fester Random Seed (42) für Clustering und t-SNE
- 99 automatisierte Tests als Regressionssicherung
