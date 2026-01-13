"""Prompt templates for LLM."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

SYSTEM_PROMPT = """Du bist ein Controlling-Assistent für Abweichungsanalysen.

STRIKTE REGELN:
1. KEINE BERECHNUNGEN - Alle Zahlen sind bereits berechnet. Verwende NUR die gegebenen Werte.
2. KEINE ERFUNDENEN GRÜNDE - Wenn die Ursache unklar ist:
   - Kennzeichne als "Hypothese" oder "Indiz"
   - Oder formuliere als offene Frage
3. EVIDENZ-LABELS für jede Aussage:
   - "Datenbasiert": Direkt aus Daten ableitbar
   - "Indiz": Plausible Vermutung
   - "Offen": Muss geklärt werden
4. Antworte auf Deutsch.

OUTPUT (nur JSON, kein anderer Text):
{
  "headline": "Kurze Zusammenfassung (max 10 Worte)",
  "summary": ["Bullet 1", "Bullet 2", "Bullet 3"],
  "drivers": [{"name": "...", "delta": 123, "share": 0.45}],
  "evidence": [{"label": "Datenbasiert|Indiz|Offen", "text": "..."}],
  "questions": ["Offene Frage 1", "..."]
}
"""


def format_context(
    account: str,
    account_name: str,
    prior: float,
    current: float,
    delta: float,
    delta_pct: float,
    drivers: List[Dict[str, Any]],
    samples: List[Dict[str, Any]],
    keywords: List[Tuple[str, int]],
) -> str:
    """Format analysis context for LLM prompt.

    Args:
        account: Account number
        account_name: Account description
        prior: Prior period sum
        current: Current period sum
        delta: Variance
        delta_pct: Variance percentage
        drivers: Top drivers list
        samples: Sample postings
        keywords: Top keywords

    Returns:
        Formatted prompt string
    """
    pct_str = f"{delta_pct:+.1%}" if delta_pct is not None else "n/a"

    drivers_text = "\n".join(
        f"- {d.get('cost_center') or d.get('vendor', 'n/a')}: "
        f"VJ {d.get('prior', 0):,.0f} → AQ {d.get('current', 0):,.0f} = "
        f"Delta {d.get('delta', 0):+,.0f} ({d.get('share', 0):.0%})"
        for d in drivers
    ) or "Keine Treiber"

    samples_text = "\n".join(
        f"- {s.get('posting_date', '')}: {s.get('amount', 0):+,.0f} | "
        f"{str(s.get('text', ''))[:40]}"
        for s in samples[:5]
    ) or "Keine Buchungen"

    kw_text = ", ".join(f"{k}({c})" for k, c in keywords[:8]) or "keine"

    return f"""KONTO: {account} - {account_name}

VARIANZ:
- Vorjahr: {prior:,.0f} EUR
- Aktuell: {current:,.0f} EUR
- Delta: {delta:+,.0f} EUR ({pct_str})

TREIBER:
{drivers_text}

TOP BUCHUNGEN:
{samples_text}

KEYWORDS: {kw_text}

Erstelle einen Management-Kommentar.
"""
