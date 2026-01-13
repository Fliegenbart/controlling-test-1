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
streamlit run app/app.py
```

Öffnet http://localhost:8501.

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
├── app/app.py               # Streamlit UI
├── variance_copilot/
│   ├── io.py                # CSV Reader
│   ├── normalize.py         # Spalten-Mapping
│   ├── variance.py          # Variance Engine
│   ├── keywords.py          # Text-Analyse
│   ├── ollama_client.py     # Ollama API
│   └── prompts.py           # LLM Prompts
├── tests/
├── sample_data/
└── pyproject.toml
```

## Hard Constraints

- Daten verlassen niemals die Maschine
- LLM rechnet nicht - alle Zahlen deterministisch
- Keine erfundenen Gründe - Evidenz-Labels: Datenbasiert/Indiz/Offen
