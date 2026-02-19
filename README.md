# SBERT Social-Media-Post Analyse

Sentence-BERT Pipeline für Similarity Search, Clustering und Evaluation von Social-Media-Posts.

## Projektstruktur

```
├── configs/
│   └── config.yaml          # Zentrale Konfiguration
├── data/
│   ├── raw/
│   │   └── sample_posts.csv  # Beispiel-Dataset (~43 Posts, 6 Kategorien)
│   └── processed/
├── results/                   # Outputs der Pipeline
├── scripts/
│   └── run_pipeline.py        # CLI Entry Point
├── src/
│   ├── preprocess.py          # Text laden & bereinigen
│   ├── embed.py               # Sentence-BERT Embeddings + Caching
│   ├── similarity.py          # Cosine-Similarity Top-k Suche
│   ├── cluster.py             # KMeans, DBSCAN (+ Grid Search), HDBSCAN
│   ├── evaluate.py            # Metriken + TF-IDF Keywords + Weak-Label Eval
│   └── report.py              # JSON/CSV/Markdown Reports
├── requirements.txt
└── README.md
```

## Setup

**Voraussetzung:** Python 3.11+

```bash
# 1. Virtual Environment erstellen
python3 -m venv .venv
source .venv/bin/activate

# 2. Dependencies installieren
pip install -r requirements.txt
```

## Quickstart

```bash
# Pipeline ausführen (Default-Config)
python scripts/run_pipeline.py --config configs/config.yaml
```

### Outputs

| Datei | Beschreibung |
|---|---|
| `results/metrics.json` | Silhouette, Davies-Bouldin, NMI, ARI, Purity pro Methode |
| `results/clusters.csv` | Post-IDs mit Cluster-Zuordnungen aller Methoden |
| `results/examples.md` | Top-Keywords (Uni-/Bigrams) und repräsentative Beispiele pro Cluster |
| `results/dbscan_grid.csv` | DBSCAN Grid-Search-Ergebnisse (alle Kombinationen) |
| `results/embeddings.npy` | Gecachte Embeddings (werden beim nächsten Lauf wiederverwendet) |

## Konfiguration

Alle Parameter werden über `configs/config.yaml` gesteuert:

- **model.name** – Sentence-Transformer Modellname (Default: `all-MiniLM-L6-v2`)
- **preprocessing** – URL/Mention-Entfernung, Lowercase, Min-Textlänge
- **clustering.methods** – Liste von Methoden mit Parametern
- **similarity.query_index** – Index des Abfrage-Posts für die Top-k Suche
- **data.language_filter** – Optional: z.B. `"en"` (benötigt `pip install langdetect`)
- **data.category_column** – Name der Weak-Label-Spalte (aktiviert NMI/ARI/Purity)

## DBSCAN Parameter-Tuning (Grid Search)

Statt feste DBSCAN-Parameter zu wählen, kann eine automatische Grid Search konfiguriert werden.
In `config.yaml` unter der DBSCAN-Methode:

```yaml
- name: "dbscan"
  params:
    metric: "cosine"
  grid_search:
    eps_values: [0.15, 0.20, 0.25, 0.30, 0.35]
    min_samples_values: [3, 4, 5]
```

### Wie funktioniert es?

1. Alle Kombinationen aus `eps_values × min_samples_values` werden durchprobiert.
2. Für jede Kombination werden Silhouette-Score, Davies-Bouldin, Cluster-Anzahl und Noise-Punkte berechnet.
3. Die **beste Kombination** wird automatisch gewählt (höchster Silhouette; Tie-Breaker: weniger Noise, dann moderatere Cluster-Anzahl).
4. Diese Best-Kombi wird als DBSCAN-Ergebnis in `clusters.csv` und `examples.md` verwendet.

### `results/dbscan_grid.csv` lesen

| Spalte | Bedeutung |
|---|---|
| `eps` | Getesteter eps-Wert |
| `min_samples` | Getesteter min_samples-Wert |
| `n_clusters` | Anzahl gefundener Cluster (ohne Noise) |
| `n_noise` | Anzahl als Noise klassifizierter Punkte |
| `silhouette` | Silhouette-Score (null wenn < 2 Cluster) |
| `davies_bouldin` | Davies-Bouldin-Index (null wenn < 2 Cluster) |

Ohne `grid_search`-Block verwendet die Pipeline die festen `params` wie bisher.

## Weak-Label Evaluation

Falls die Daten eine `category`-Spalte enthalten (konfigurierbar via `data.category_column`),
berechnet die Pipeline automatisch:

| Metrik | Beschreibung |
|---|---|
| **NMI** | Normalized Mutual Information zwischen Clustering und Kategorien |
| **ARI** | Adjusted Rand Index |
| **Purity** | Anteil korrekt zugeordneter Punkte (Noise ignoriert) |
| **Purity (inkl. Noise)** | Wie Purity, aber Noise-Punkte als eigene Gruppe |

Diese Metriken erscheinen unter `weak_label` pro Methode in `results/metrics.json`.

## Modell austauschen

In `config.yaml` einfach den Modellnamen ändern und den Embedding-Cache löschen:

```bash
rm results/embeddings.npy results/embedding_ids.json
# config.yaml anpassen, z.B.:
# model.name: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
python scripts/run_pipeline.py
```

## Reproduzierbarkeit

- Fester Random Seed (konfigurierbar via `seed` in config.yaml)
- Embeddings werden gecacht (`results/embeddings.npy` + `results/embedding_ids.json`)
- Bei gleichen Daten + gleichem Modell → identische Ergebnisse
