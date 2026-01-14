"""Prompt templates for LLM."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# --- STRICT Prompt (default) ---
SYSTEM_PROMPT_STRICT = """Du bist ein Controlling-Assistent für Abweichungsanalysen.

STRIKTE REGELN (MÜSSEN eingehalten werden):

1. KEINE ZAHLEN ERFINDEN, NICHT RECHNEN
   - Alle Zahlen sind bereits berechnet und werden dir gegeben.
   - Verwende NUR die exakten Werte aus den Daten.
   - Runde NICHT, addiere NICHT, schätze NICHT.

2. KEINE GRÜNDE ERFINDEN
   - Wenn die Ursache unklar ist: als "Hypothese" markieren ODER als offene Frage formulieren.
   - Sage NIE "wahrscheinlich weil..." ohne Datenbeleg.

3. EVIDENZ-LABELS (PFLICHT für jeden Bullet)
   - [Datenbasiert]: Direkt aus den Daten ableitbar (z.B. "Top-Buchung zeigt...")
   - [Indiz]: Plausible Vermutung basierend auf Mustern (z.B. "Keyword 'Sonder' deutet auf...")
   - [Offen]: Muss beim Fachbereich geklärt werden

4. OUTPUT: NUR GÜLTIGES JSON
   - Kein Markdown, kein Zusatztext vor oder nach dem JSON.
   - Keine Erklärungen, keine Einleitung.

JSON-SCHEMA (exakt einhalten):
{
  "headline": "Max 10 Worte Zusammenfassung",
  "summary": [
    "[Datenbasiert] Aussage 1...",
    "[Indiz] Aussage 2...",
    "[Offen] Aussage 3..."
  ],
  "drivers": [
    {"name": "Treiber-Name", "delta": 12345, "share": 0.45}
  ],
  "evidence": [
    {"label": "Datenbasiert", "text": "Begründung..."},
    {"label": "Indiz", "text": "Vermutung..."},
    {"label": "Offen", "text": "Frage an Fachbereich..."}
  ],
  "questions": ["Offene Frage 1", "Offene Frage 2"]
}
"""

# --- NORMAL Prompt (less strict) ---
SYSTEM_PROMPT_NORMAL = """Du bist ein Controlling-Assistent für Abweichungsanalysen.

REGELN:
1. Verwende die gegebenen Zahlen, rechne nicht selbst.
2. Unterscheide zwischen Fakten und Vermutungen.
3. Antworte auf Deutsch.

OUTPUT als JSON:
{
  "headline": "Kurze Zusammenfassung",
  "summary": ["Bullet 1", "Bullet 2", "Bullet 3"],
  "drivers": [{"name": "...", "delta": 123, "share": 0.45}],
  "evidence": [{"label": "Datenbasiert|Indiz|Offen", "text": "..."}],
  "questions": ["Offene Frage 1"]
}
"""

# Legacy alias
SYSTEM_PROMPT = SYSTEM_PROMPT_STRICT


def get_system_prompt(mode: str = "strict") -> str:
    """Get system prompt by mode.

    Args:
        mode: "strict" or "normal"

    Returns:
        System prompt string
    """
    if mode == "normal":
        return SYSTEM_PROMPT_NORMAL
    return SYSTEM_PROMPT_STRICT


def compute_oneoff_indicators(samples: List[Dict[str, Any]], total_delta: float) -> Dict[str, Any]:
    """Compute one-off indicators from samples.

    Args:
        samples: Sample postings sorted by abs(amount)
        total_delta: Total delta for the account

    Returns:
        Dict with top1_share, top5_share, top1_doc
    """
    if not samples or abs(total_delta) < 0.01:
        return {"top1_share": 0.0, "top5_share": 0.0, "top1_doc": None}

    abs_total = abs(total_delta)
    top1_amt = abs(samples[0].get("amount", 0)) if samples else 0
    top5_amt = sum(abs(s.get("amount", 0)) for s in samples[:5])

    return {
        "top1_share": top1_amt / abs_total if abs_total > 0 else 0,
        "top5_share": top5_amt / abs_total if abs_total > 0 else 0,
        "top1_doc": samples[0].get("document_no") if samples else None,
    }


def format_context(
    account: str,
    account_name: str,
    prior: float,
    current: float,
    delta: float,
    delta_pct: Optional[float],
    drivers: List[Dict[str, Any]],
    samples: List[Dict[str, Any]],
    keywords: List[Tuple[str, int]],
    abs_delta: Optional[float] = None,
    share_of_total: Optional[float] = None,
) -> str:
    """Format analysis context for LLM prompt.

    Provides condensed data:
    - Key metrics (prior, current, delta, delta_pct)
    - Top drivers table
    - One-off indicators (Top1/Top5 doc share)
    - Keywords
    - Max 8 sample rows

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
        abs_delta: Absolute delta (optional)
        share_of_total: Share of total abs delta (optional)

    Returns:
        Formatted prompt string
    """
    pct_str = f"{delta_pct:+.1%}" if delta_pct is not None else "n/a"

    # One-off indicators
    oneoff = compute_oneoff_indicators(samples, delta)
    oneoff_text = (
        f"Top1-Anteil: {oneoff['top1_share']:.0%}, "
        f"Top5-Anteil: {oneoff['top5_share']:.0%}"
    )
    if oneoff["top1_doc"]:
        oneoff_text += f", Top1-Dok: {oneoff['top1_doc']}"

    # Drivers table (condensed)
    drivers_text = "\n".join(
        f"  {d.get('cost_center') or d.get('vendor', 'n/a')}: "
        f"Δ {d.get('delta', 0):+,.0f} ({d.get('share', 0):.0%})"
        for d in drivers[:5]
    ) or "  Keine Treiber"

    # Samples (max 8, condensed)
    samples_text = "\n".join(
        f"  {s.get('posting_date', '')}: {s.get('amount', 0):+,.0f} | "
        f"{str(s.get('text', ''))[:35]}"
        for s in samples[:8]
    ) or "  Keine Buchungen"

    # Keywords (max 8)
    kw_text = ", ".join(f"{k}({c})" for k, c in keywords[:8]) or "keine"

    # Optional extended metrics
    extra_metrics = ""
    if abs_delta is not None:
        extra_metrics += f"\n- |Delta|: {abs_delta:,.0f} EUR"
    if share_of_total is not None:
        extra_metrics += f"\n- Anteil Gesamt: {share_of_total:.1%}"

    return f"""KONTO: {account} - {account_name}

KENNZAHLEN:
- Vorjahr (VJ): {prior:,.0f} EUR
- Aktuell (AQ): {current:,.0f} EUR
- Delta: {delta:+,.0f} EUR ({pct_str}){extra_metrics}

ONE-OFF INDIKATOREN:
{oneoff_text}

TREIBER (Top 5):
{drivers_text}

BUCHUNGEN (Top 8):
{samples_text}

KEYWORDS: {kw_text}

Analysiere diese Varianz. Antworte NUR mit JSON.
"""


# --- EXECUTIVE SUMMARY Prompt ---
SYSTEM_PROMPT_EXECUTIVE = """Du bist ein Senior Controller, der einen Executive Summary für die Geschäftsführung erstellt.

STRIKTE REGELN:
1. Verwende NUR die gegebenen Zahlen - erfinde nichts.
2. Fokussiere auf die TOP 5-7 wichtigsten Abweichungen.
3. Erkläre Muster und Zusammenhänge zwischen Konten.
4. Unterscheide zwischen Fakten, Indizien und offenen Punkten.
5. Formuliere konkrete Empfehlungen und nächste Schritte.

OUTPUT: NUR GÜLTIGES JSON (kein Markdown, kein Text davor/danach):
{
  "headline": "Executive Summary in einem Satz (max 15 Worte)",
  "key_findings": [
    "Wichtigste Erkenntnis 1 (mit Zahlen belegt)",
    "Wichtigste Erkenntnis 2",
    "Wichtigste Erkenntnis 3"
  ],
  "top_variances": [
    {"name": "Kontoname", "delta": 12345, "reason": "Kurze Erklärung"},
    {"name": "Kontoname 2", "delta": -5000, "reason": "Kurze Erklärung"}
  ],
  "patterns": [
    "Erkanntes Muster 1 (z.B. 'Mehrere Konten zeigen Kostensteigerungen bei Lieferant X')",
    "Erkanntes Muster 2"
  ],
  "recommendations": [
    "Konkrete Handlungsempfehlung 1",
    "Konkrete Handlungsempfehlung 2"
  ],
  "open_items": [
    "Offener Punkt der geklärt werden muss 1",
    "Offener Punkt 2"
  ]
}
"""


def format_executive_context(
    period_info: str,
    total_prior: float,
    total_current: float,
    total_delta: float,
    variance_summary: List[Dict[str, Any]],
    account_details: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Format context for executive summary generation.

    Args:
        period_info: Period description (e.g., "Q2 2024 vs Q2 2025")
        total_prior: Total prior period value
        total_current: Total current period value
        total_delta: Total variance
        variance_summary: List of material variances with account info
        account_details: Optional additional details per account

    Returns:
        Formatted prompt string
    """
    delta_pct = (total_delta / total_prior * 100) if total_prior != 0 else 0

    # Format variance table
    variance_lines = []
    for v in variance_summary[:15]:  # Top 15 accounts
        acc = v.get("account", "")
        name = v.get("account_name", "")[:30]
        prior = v.get("prior", 0)
        current = v.get("current", 0)
        delta = v.get("delta", 0)
        pct = v.get("delta_pct")
        pct_str = f"{pct:+.1%}" if pct is not None else "-"
        share = v.get("share_of_total_abs_delta", 0)

        variance_lines.append(
            f"  {acc} {name}: VJ {prior:,.0f} → AQ {current:,.0f} | "
            f"Δ {delta:+,.0f} ({pct_str}) | Anteil: {share:.1%}"
        )

    variance_table = "\n".join(variance_lines)

    # Format account details if provided
    details_text = ""
    if account_details:
        details_lines = []
        for detail in account_details[:10]:
            acc = detail.get("account", "")
            name = detail.get("account_name", "")
            drivers = detail.get("top_drivers", [])
            keywords = detail.get("keywords", [])

            driver_str = ", ".join(
                f"{d.get('name', '')}: {d.get('delta', 0):+,.0f}"
                for d in drivers[:3]
            ) if drivers else "keine"

            kw_str = ", ".join(keywords[:5]) if keywords else "keine"

            details_lines.append(
                f"  {acc} {name}:\n"
                f"    Treiber: {driver_str}\n"
                f"    Keywords: {kw_str}"
            )

        details_text = "\n\nDETAILS PRO KONTO:\n" + "\n".join(details_lines)

    return f"""EXECUTIVE SUMMARY ANFRAGE

BERICHTSZEITRAUM: {period_info}

GESAMTÜBERSICHT:
- Vorjahr Gesamt: {total_prior:,.0f} EUR
- Aktuell Gesamt: {total_current:,.0f} EUR
- Gesamtvarianz: {total_delta:+,.0f} EUR ({delta_pct:+.1f}%)
- Anzahl materieller Abweichungen: {len(variance_summary)}

MATERIELLE ABWEICHUNGEN (sortiert nach Betrag):
{variance_table}
{details_text}

Erstelle einen Executive Summary. Fokussiere auf:
1. Die wichtigsten Abweichungen und deren Ursachen
2. Erkannte Muster über mehrere Konten
3. Konkrete Handlungsempfehlungen
4. Offene Punkte die geklärt werden müssen

Antworte NUR mit JSON.
"""
