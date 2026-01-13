# Variance Copilot

Lokales Controlling-Tool für Quartals-Abweichungsanalyse (YoY) mit KI-Kommentierung via Ollama.

## Setup

```bash
# 1. Virtual Environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Dependencies
pip install -e ".[dev]"

# 3. Ollama starten (in separatem Terminal)
ollama serve

# 4. Optional: Modell herunterladen
ollama pull llama3.2
```

## App starten

```bash
streamlit run streamlit_app.py
```

Öffnet http://localhost:8501.

## Demo-Modus

Die App enthält einen Demo-Loader für schnelles Testen:

1. **Sidebar öffnen** (Pfeil oben links)
2. **"Load Sample Data"** Button klicken
3. Die synthetischen Daten (Q2 2024 vs Q2 2025) werden automatisch geladen

Alternativ: Eigene CSVs über den normalen Upload-Flow hochladen.

**Reset:** Über den "Reset" Button können die Demo-Daten zurückgesetzt werden.

## Sample-Daten

Im Ordner `sample_data/` liegen synthetische Testdaten:

| Datei | Zeitraum | Zeilen |
|-------|----------|--------|
| `buchungen_Q2_2024_fiktiv.csv` | April–Juni 2024 | ~370 |
| `buchungen_Q2_2025_fiktiv.csv` | April–Juni 2025 | ~400 |

**Eingebaute Effekte in 2025:**
- **Account 5000** (Material Reagenzien): Höhere Kosten bei Roche und Siemens Healthineers
- **Account 6200** (Wartung): One-off Buchung -120.000 EUR (Dokument BK999999)
- **Account 4000** (Umsatzerlöse): ~10-15% mehr Umsatz als 2024

**Konvention:** Umsatz positiv, Aufwand negativ.

### Daten neu generieren

```bash
python scripts/generate_sample_data.py
```

Das Skript erzeugt reproduzierbare Daten (seed=42).

## Tests

```bash
pytest -v
```

## Konfiguration (.env)

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
OLLAMA_TIMEOUT=120
```

## Projektstruktur

```
variance_copilot/
├── streamlit_app.py           # Streamlit UI (Entry Point)
├── variance_copilot/
│   ├── io.py                  # CSV Reader
│   ├── normalize.py           # Spalten-Mapping
│   ├── variance.py            # Variance Engine + Materiality
│   ├── keywords.py            # Text-Analyse
│   ├── ollama_client.py       # Ollama API + JSON Extraction
│   └── prompts.py             # LLM Prompts (strict/normal)
├── scripts/
│   └── generate_sample_data.py  # Sample-Daten Generator
├── sample_data/
│   ├── buchungen_Q2_2024_fiktiv.csv
│   └── buchungen_Q2_2025_fiktiv.csv
├── tests/
│   ├── test_variance.py       # Materiality Tests
│   ├── test_ollama_client.py  # JSON Extraction Tests
│   └── ...
├── pyproject.toml
└── README.md
```

## Hard Constraints

- Daten verlassen niemals die Maschine
- LLM rechnet nicht - alle Zahlen deterministisch
- Keine erfundenen Gründe - Evidenz-Labels: Datenbasiert/Indiz/Offen
