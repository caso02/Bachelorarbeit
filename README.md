# Social AI — Community OODA-Loop (ZHAW Bachelorarbeit 2026)

Autonomes Multi-Agent-System zur semantischen Analyse von Social-Media-Posts und ethisch gestalteter Community-Vernetzung.

## Architektur

Das System implementiert einen **OODA-Loop** (Observe-Orient-Decide-Act) als CrewAI-inspiriertes Multi-Agent-Framework:

```
Observe  → Collector Agent     Daten sammeln (Reddit / Kaggle / Mock)
Orient   → Analysis Agent      Gemini Embeddings, Topic-Routing, Clustering, Readiness-Scores
Decide   → Action Agent        Ethische Vernetzungsvorschläge (Nudges) generieren
Act      → Ethics Agent        Responsible-AI-Prüfung → APPROVED / REVISED / REJECTED
```

Die Orchestrierung erfolgt über eine leichtgewichtige, dependency-freie Abstraktion (`Agent`, `Task`, `Crew`), die CrewAI-Konzepte nachbildet (CrewAI 0.11.2 ist nicht mit Python 3.14 kompatibel).

### Value-Sensitive Design (VSD)

- **Unaufdringlich** — keine FOMO-Sprache, kein Druck
- **Transparent** — jeder Vorschlag enthält eine ehrliche Begründung
- **Autonomieerhaltend** — Nutzer entscheidet frei
- **Privatsphäre** — Usernamen werden SHA-256-pseudonymisiert, nie im Klartext gespeichert

## Projektstruktur

```
├── configs/
│   └── config.yaml              # Zentrale Konfiguration
├── data/
│   ├── raw/sample_posts_300.csv  # Mock-Dataset (300 Posts, 6 Kategorien)
│   └── kaggle/                   # Optional: Kaggle Reddit CSV
├── results/                      # Pipeline-Outputs (JSON, CSV, Plots)
├── scripts/
│   └── run_pipeline.py           # Klassische Analyse-Pipeline (CLI)
├── src/
│   ├── community_crew.py         # OODA-Loop Orchestrierung
│   ├── app.py                    # Streamlit Dashboard (OODA-Visualisierung + Nudge-Review)
│   ├── universal_collector.py    # Datensammlung (Reddit → Kaggle → Mock)
│   ├── embed.py                  # Gemini / SBERT Embeddings (Batch, Cache)
│   ├── routing.py                # Topic-Routing via Prototyp-Embeddings
│   ├── hierarchical.py           # HDBSCAN / KMeans Subclustering
│   ├── action_agent.py           # Nudge-Generierung (Gemini LLM + VSD)
│   ├── ethics_agent.py           # Responsible-AI-Prüfung (APPROVED / REVISED / REJECTED)
│   ├── agent_state.py            # In-Memory Inter-Agent-Kommunikation
│   ├── utils.py                  # Shared Utilities (.env, JSON-Parsing, Client-Cache)
│   ├── evaluate.py               # Metriken, TF-IDF, MRL-Truncation
│   ├── cluster.py                # KMeans, DBSCAN, HDBSCAN
│   ├── report.py                 # JSON/CSV/Markdown Reports
│   └── visualize.py              # t-SNE Plots
├── tests/                        # Unit-Tests (pytest)
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

Öffnet ein interaktives Dashboard mit:
- Keyword-Suche für Topic-Filterung
- Schrittweiser OODA-Loop Ausführung mit Live-Status
- **Community Nudges** mit Ethics-Badge (APPROVED / REVISED / REJECTED)
  - Tab "Begründung & Ethik": VSD-Erklärung + Responsible-AI-Reasoning
  - Tab "Analysierte Posts": alle Posts des Clusters mit Readiness-Kennzeichnung (🔥 vernetzungsbereit)
  - Cluster-Stärke: Gesamtposts vs. hochgradig vernetzungsbereite Posts
- **Technical Analytics**: t-SNE Cluster-Plot, Readiness-Heatmap, Cluster-Tabelle

### CLI (OODA-Loop)

```bash
python src/community_crew.py --query "health"
```

### Klassische Analyse-Pipeline

```bash
python scripts/run_pipeline.py --config configs/config.yaml
```

## Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

## Datenquellen (Prioritäts-Fallback)

| Modus | Bedingung | Beschreibung |
|-------|-----------|--------------|
| 🟢 Reddit API | `REDDIT_CLIENT_ID` in `.env` | Live-Suche via praw |
| 🟡 Kaggle CSV | `data/kaggle/reddit_data.csv` vorhanden | Statische Reddit-Daten |
| 🔴 Mock-Fallback | Immer verfügbar | 300 synthetische Posts |

## Pipeline-Outputs

| Datei | Beschreibung |
|-------|--------------|
| `results/crew_posts.json` | Gesammelte Posts mit Metadaten |
| `results/crew_analysis.json` | Cluster mit Readiness-Scores |
| `results/hierarchy.csv` | Post-Cluster-Zuordnungen |
| `results/crew_nudges.json` | Generierte Nudges (vor Ethics-Review) |
| `results/final_nudges.json` | Freigegebene Nudges (nur APPROVED/REVISED) |
| `results/ethics_reviews/` | Einzelne Ethics-Review-Protokolle |
| `results/cluster_plot.png` | t-SNE Visualisierung |

## Konfiguration

Alle Parameter werden über `configs/config.yaml` gesteuert:

- **model.name** — Embedding-Modell (`gemini-embedding-2-preview` oder SBERT)
- **interest_labels** — Topic-Kategorien mit Seed-Sätzen für Routing
- **hierarchical** — HDBSCAN/KMeans Parameter für Subclustering
- **preprocessing** — Text-Bereinigung, Min-Textlänge
- **output** — Pfade für Cache und Ergebnisse

## Reproduzierbarkeit

- Embeddings werden gecacht (`results/embeddings.npy` + `results/embedding_ids.json`)
- Label-Prototypen gecacht (`results/proto_cache/`)
- Fester Random Seed für Clustering und t-SNE
