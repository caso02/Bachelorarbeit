# Recon-Report: Query-Modus für Streamlit-Frontend

## 1. Frontend-Status quo

### Dateien

Eine einzige Streamlit-Datei: `src/app.py` (~424 Zeilen).

### Seiten/Tabs/Bereiche

```
Sidebar
├── "Methodik" Header
├── Datenquelle (Status: Reddit 🟢 / Kaggle 🟡 / Mock 🔴)
├── Value-Sensitive Design (Info-Text)
├── Datenschutz (Info-Text)
└── Agenten-Architektur (Info-Text)

Hauptbereich
├── Header: "Social AI Dashboard"
├── Input-Zeile (2 Spalten)
│   ├── st.text_input("Keyword / Thema")   ← einziger User-Input
│   └── st.button("🚀 Analyse starten")
├── st.status("OODA-Loop läuft...")         ← Phasen-Callbacks
└── Ergebnis-Tabs
    ├── Tab 1: "🚀 Community Nudges"
    │   ├── Metriken (geprüft / freigegeben / abgelehnt)
    │   └── Pro Nudge:
    │       ├── Sub-Tab "Begründung & Ethik"
    │       └── Sub-Tab "Analysierte Posts"
    └── Tab 2: "📊 Technical Analytics"
        ├── Metriken (Posts / Cluster / Readiness)
        ├── t-SNE-Plot (cluster_plot.png)
        └── Cluster-Tabelle
```

### User-Inputs (vollständige Liste)

| Element | Typ | Variable | Zweck |
|---|---|---|---|
| Keyword / Thema | `st.text_input` | `query` | Filtert Posts nach Suchbegriff; leer = alle Posts |
| Analyse starten | `st.button` | `run_btn` | Triggert Pipeline |

Keine weiteren Inputs — kein Subreddit-Filter, kein Limit-Slider, keine Modell-Auswahl im Frontend.

### Wie liest das Frontend Ergebnisse ein?

**Hybrid-Ansatz:** Das Frontend triggert die Pipeline direkt UND liest danach Dateien.

1. **Pipeline-Trigger** (Zeile 132–152):
   ```python
   state = AgentState(query=query)
   state = coordinator.run(state, on_phase=_on_phase)
   ```
   Die App erstellt einen `AgentState`, ruft `coordinator.run()` auf und erhält den mutierten State zurück.

2. **Ergebnis-Zugriff** — sowohl über State als auch über Dateien:
   - `state.final_nudges` → Nudge-Anzeige (Zeile 160)
   - `state.messages` → Audit Trail (Zeile 169)
   - `RESULTS_DIR / "final_nudges.json"` → Nudge-Details (Zeile 193)
   - `RESULTS_DIR / "crew_posts.json"` → Post-Daten (Zeile 208)
   - `RESULTS_DIR / "hierarchy.csv"` → Cluster-Zuordnungen (Zeile 209)
   - `RESULTS_DIR / "crew_analysis.json"` → Analyse-Metriken (Zeile 210)
   - `RESULTS_DIR / "cluster_plot.png"` → t-SNE-Visualisierung (Zeile 381)

### Triggert das Frontend die Pipeline?

**Ja, direkt.** Beim Klick auf "Analyse starten" wird `coordinator.run()` synchron aufgerufen. Die Phasen-Callbacks (`_on_phase`) aktualisieren die Streamlit-UI in Echtzeit.

---

## 2. Pipeline-Einstiegspunkte

### Terminal-Befehl

```bash
# Hierarchical Pipeline (Standard)
python scripts/run_pipeline.py --config configs/config.yaml --mode hierarchical

# Mit Stabilität
python scripts/run_pipeline.py --config configs/config.yaml --stability

# 3-Backend-Vergleich (SBERT / Gemini 3072 / Gemini 768 MRL)
python scripts/run_pipeline.py --config configs/config.yaml --eval-compare

# Erweiterte Evaluation (8 Kategorien, Evolution-AI)
python scripts/run_evaluation.py --run 1
```

### Primärer Entry-Point

**CLI:** `scripts/run_pipeline.py` → `main()` → `run_hierarchical(cfg)` oder `_run_eval_compare(cfg)`

**Streamlit:** `src/app.py` → `coordinator.run(state, on_phase=_on_phase)`

### Datenfluss

```
Rohdaten              Embeddings                    Routing + Clustering           Nudges + Ethics
─────────             ──────────                    ────────────────────           ───────────────
CSV/Reddit/Mock  →  load_or_compute_embeddings()  →  embed_label_prototypes()   →  ActionAgent
                    ↓ Cache: results/embeddings.npy   route_posts()                  .generate_community_nudge()
                    ↓ Cache: results/embedding_ids.json  cluster_per_topic()       →  EthicsAgent
                    ↓                                  calculate_readiness_score()     .review_nudge()
                    ↓                                  ↓                             ↓
                    results/*.npy                   results/hierarchy.csv          results/final_nudges.json
                                                   results/routing.csv            results/ethics_reviews/*.json
                                                   results/metrics.json           results/agent_messages.json
```

### Embedding-Persistierung

Die Embeddings der 49'266 Beiträge sind **nicht** vollständig vorberechnet. Die vorhandenen Caches:

| Pfad | Inhalt | Posts |
|---|---|---|
| `results/embeddings.npy` | Gemini 3072 | 300 (Pilotstudie) |
| `results/eval_compare/sbert/embeddings.npy` | SBERT 384 | 300 |
| `results/eval_compare/gemini_3072/embeddings.npy` | Gemini 3072 | 300 |
| `results/run1_reddit_8cat/gemini_3072/embeddings.npy` | Gemini 3072 | 700 (Evolution-AI) |
| `results/run1_reddit_8cat/sbert/embeddings.npy` | SBERT 384 | 700 |

Für den Query-Modus auf dem vollen 49k-Datensatz müssten die Embeddings erst berechnet und persistiert werden (~$5–10 Gemini-API-Kosten für 49k Posts).

---

## 3. Architektur-Anschlusspunkte

### Wo einen Query-Filter einhängen?

**Sauberste Stelle: Zwischen Collect und Analyse, als neuer "Retrieve"-Schritt.**

Begründung:
- Der Collector liefert Rohdaten (Posts). Der Analyst erwartet Posts + Embeddings.
- Ein Retrieve-Schritt würde: (1) die Query embedden, (2) per Kosinus-Ähnlichkeit die Top-N relevantesten Posts aus dem vorberechneten Korpus-Index selektieren, (3) nur diese an den Analyst weiterreichen.
- Das lässt den Collector (Daten laden) und den Analyst (Routing/Clustering) unverändert.

### Wiederverwendbare Module

| Modul | Funktion | Wiederverwendung für Query-Modus |
|---|---|---|
| `src/embed.py` | `load_or_compute_embeddings()` | Korpus-Embeddings vorberechnen + cachen |
| `src/embed.py` | `_embed_with_gemini()` / `_embed_with_sbert()` | Query-Embedding berechnen |
| `src/similarity.py` | `find_top_k()` | **Fast direkt nutzbar** — braucht nur eine Variante die einen Query-Vektor statt einen Index akzeptiert |
| `src/routing.py` | `route_posts()` + `embed_label_prototypes()` | Unverändert auf dem gefilterten Subset |
| `src/hierarchical.py` | `cluster_per_topic()` | Unverändert |
| `src/community_crew.py` | `_analyse_stateful()`, `_generate_nudges_stateful()`, Coordinator | Unverändert |

### Vektor-Datenbank-Status

**Keine vorhanden.** Kein FAISS, ChromaDB, Annoy oder HNSWLIB im Projekt. Alle Similarity-Operationen nutzen `sklearn.metrics.pairwise.cosine_similarity` in-memory.

Für 49k Posts × 768 Dimensionen = ~150 MB als float32. Das passt problemlos in den RAM für brute-force Cosine-Similarity. Ein FAISS-Index wäre nice-to-have (Sub-Millisekunden statt ~50ms), aber nicht zwingend nötig.

### Relevante Config-Felder

| Feld | Nutzung im Query-Modus |
|---|---|
| `model.name` | Embedding-Modell für Query + Korpus (muss identisch sein) |
| `model.gemini.output_dimensionality` | Dimensionalität (768 für MRL) |
| `interest_labels` | Routing-Prototypen (unverändert) |
| `hierarchical.*` | Subclustering-Parameter (unverändert) |
| `data.input_path` | Könnte auf den vollen Kaggle-Datensatz zeigen statt auf die 300-Post-Stichprobe |

**Neue Config-Felder nötig:**
- `query_mode.corpus_embeddings_path` — Pfad zum vorberechneten Korpus-Index
- `query_mode.top_k` — Anzahl zu retrievender Posts (default: 200)
- `query_mode.min_similarity` — Mindest-Ähnlichkeit (optional, für OOD-Filterung)

---

## 4. Risiken / kritische Stellen

### Tests die brechen könnten

| Test | Risiko | Grund |
|---|---|---|
| `test_crew.py::TestCoordinator` (11 Tests) | **Mittel** | Wenn der Coordinator-Flow um einen Retrieve-Schritt erweitert wird, müssen die Mocks angepasst werden |
| `test_collector.py::TestKaggleMultiCSV` | **Niedrig** | Nur wenn der Collector-Code selbst geändert wird (sollte nicht nötig sein) |
| `test_routing.py` | **Niedrig** | Routing bleibt unverändert — arbeitet auf beliebigem Embedding-Input |

**Kein Test** prüft aktuell den End-to-End-Flow über den Streamlit-Entry-Point. Das Frontend ist untestet.

### Implizite Annahmen über Datensatzgrösse

| Stelle | Annahme | Problem bei kleinem Subset |
|---|---|---|
| `hierarchical.py`: `min_samples_for_hdbscan` | Default 6–10 | Bei Query mit <50 Treffern könnten Topics <6 Posts haben → alles fällt auf K-Means zurück |
| `community_crew.py`: Top-3-Cluster nach Readiness | Erwartet ≥3 Cluster | Bei Query-Modus mit 50–200 Posts könnten weniger als 3 Cluster entstehen |
| `evaluate.py`: `compute_metrics()` | Braucht ≥2 Non-Noise-Cluster | Bei sehr fokussiertem Query könnte alles in 1 Cluster landen |
| HDBSCAN `min_cluster_size = max(3, N//10)` | Skaliert mit N | Bei N=50 ist min_cluster_size=5, bei N=200 ist es 20 — sehr unterschiedliches Verhalten |

### Potenzielle CLI-Brüche

- **Keiner**, solange der Query-Modus als separater Codepfad implementiert wird (z.B. `coordinator.run(state)` vs. `coordinator.run_query(state, query_embedding)`).
- Risiko entsteht nur, wenn der bestehende `_collect_stateful()` Flow verändert wird.
- **Empfehlung:** Den Query-Modus als opt-in Feature implementieren (z.B. `state.mode = "query"` oder separater Entry-Point).

---

## 5. Empfehlung

### Variante A: Query-Filter VOR der Pipeline

```
User Query → Embed Query → Cosine-Search auf Korpus-Index → Top-N Posts
→ Bestehende Pipeline (Routing → Subclustering → Readiness → Nudge → Ethics)
```

**Pro:** Sauberste Trennung. Pipeline bleibt 100% unverändert. Nur ein neuer Pre-Step.
**Contra:** Doppelte Embedding-Berechnung (Korpus muss vorberechnet sein + Query wird separat embedded).

### Variante B: Neuer "Retrieve"-Schritt im Collector-Agent

```
Collector-Agent erweitert: collect() → retrieve(query_embedding, corpus_index) → gefilterte Posts
→ Rest der Pipeline unverändert
```

**Pro:** Passt in die bestehende OODA-Architektur (Observe-Phase). Collector ist der natürliche Ort für Datenfilterung.
**Contra:** Verändert bestehenden Collector-Code (Regressionsrisiko). Muss sauber zwischen Korpus-Modus und Query-Modus unterscheiden.

### Variante C: Separates Discovery-Modul

```
Neues Modul src/discovery.py:
  - Lädt Korpus-Embeddings
  - Embedded Query
  - Retrieval via Cosine-Similarity
  - Erstellt AgentState mit gefilterten Posts
  - Ruft coordinator.run(state) auf
```

**Pro:** Null Änderungen an bestehenden Modulen. Maximale Isolation. Einfach testbar. Frontend ruft Discovery statt Coordinator direkt auf.
**Contra:** Leichte Code-Duplizierung (Embedding-Berechnung). Neues Modul das gepflegt werden muss.

### Empfehlung: Variante C (Discovery-Modul)

**Begründung:**
1. **Null Risiko für bestehende Pipeline.** Kein einziges bestehendes Modul wird verändert. Die 99 Tests bleiben grün ohne Anpassung.
2. **Saubere Separation of Concerns.** Discovery (Retrieval) ist konzeptionell ein anderer Vorgang als Collection (Daten laden). Das Discovery-Modul kapselt die Vector-Search-Logik.
3. **Frontend-Integration minimal.** Die Streamlit-App braucht nur einen zweiten Button ("Query-Modus") der statt `coordinator.run()` die Discovery-Funktion aufruft, die intern wiederum `coordinator.run()` mit vorgefiltertem State aufruft.
4. **Vorberechneter Index.** Ein einmaliges Skript berechnet Embeddings für den vollen Korpus und persistiert sie. Das Discovery-Modul lädt diesen Index bei Bedarf.

### Aufwandschätzung (Variante C)

| Komponente | Aufwand | Details |
|---|---|---|
| **Backend: Korpus-Index** | 2–3h | Skript zum Vorberechnen + Persistieren aller 49k Embeddings (Gemini 768 MRL). Einmaliger API-Kostenaufwand ~$5–10. |
| **Backend: Discovery-Modul** | 3–4h | `src/discovery.py` mit `discover(query_text, top_k, corpus_path)` → AgentState. Kosinus-Ähnlichkeit auf vorberechnetem Index. Optional FAISS für Geschwindigkeit. |
| **Frontend: Query-Tab** | 2–3h | Neuer Tab oder Modus-Switch in `src/app.py`. Query-Eingabe + Top-K-Slider + "Entdecken"-Button. Ergebnis-Anzeige wiederverwendet bestehende Nudge/Analytics-Tabs. |
| **Tests** | 2h | Unit-Tests für Discovery-Modul (Query-Embedding, Top-K-Retrieval, Edge-Cases: leere Query, keine Treffer). Integrationstest: Discovery → Coordinator → Nudge. |
| **Doku** | 1h | README-Update, Config-Felder dokumentieren. |
| **Gesamt** | **10–13h** | |

### Offene Punkte (unsicher)

1. **Korpus-Grösse vs. HDBSCAN-Kalibrierung:** Wenn der Query-Modus nur 50–200 Posts zurückliefert, verhält sich HDBSCAN anders als auf 300+ Posts. Die Subclustering-Parameter (`min_cluster_size`, `min_samples`) müssten möglicherweise für den Query-Modus dynamisch angepasst werden.

2. **Cross-Lingual Readiness-Anker:** Der Readiness-Anker ist deutsch ("Ich möchte mich mit Gleichgesinnten vernetzen"), die Posts sind englisch. Für den Query-Modus wäre zu klären, ob der Query-Text ebenfalls als zweiter Readiness-Indikator dienen soll.

3. **Welcher Datensatz als Korpus?** Der alte Kaggle-Datensatz (49k Posts, 50 Subreddits) oder der Evolution-AI-Datensatz (1M Posts, 1013 Subreddits)? Letzterer bietet mehr Abdeckung, erfordert aber deutlich mehr Embedding-Kosten.

4. **Threshold für Query-Relevanz:** Ab welcher Kosinus-Ähnlichkeit zum Query ist ein Post "relevant genug"? Ein fixer Schwellenwert (z.B. 0.3) oder ein dynamischer (Top-K unabhängig vom Score)?

5. **Streaming-UX:** Die bestehende Pipeline läuft synchron in Streamlit (blockiert die UI während der Ausführung). Für den Query-Modus mit Gemini-API-Calls (Embedding + Nudge + Ethics) könnte die Wartezeit 30–60s betragen. Streaming-Feedback via `on_phase`-Callbacks ist bereits implementiert und wiederverwendbar.
