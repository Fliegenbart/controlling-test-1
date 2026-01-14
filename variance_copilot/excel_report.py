"""Excel Report Generator for Clarity."""

from __future__ import annotations

from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional

import pandas as pd


def generate_variance_excel(
    variance_df: pd.DataFrame,
    prior_total: float,
    current_total: float,
    executive_summary: Optional[Dict[str, Any]] = None,
    cost_center_summary: Optional[pd.DataFrame] = None,
    period_info: str = "Quartalsvergleich",
) -> bytes:
    """Generate Excel report with multiple sheets.

    Args:
        variance_df: Variance data by account
        prior_total: Total prior period value
        current_total: Total current period value
        executive_summary: Optional AI-generated executive summary
        cost_center_summary: Optional cost center aggregation
        period_info: Period description

    Returns:
        Excel file as bytes
    """
    output = BytesIO()

    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Sheet 1: Übersicht (Summary)
        summary_data = {
            'Kennzahl': [
                'Berichtszeitraum',
                'Vorjahr Gesamt',
                'Aktuell Gesamt',
                'Gesamtabweichung',
                'Abweichung %',
                'Anzahl Konten',
                'Generiert am',
            ],
            'Wert': [
                period_info,
                f"{prior_total:,.0f}",
                f"{current_total:,.0f}",
                f"{current_total - prior_total:+,.0f}",
                f"{((current_total - prior_total) / prior_total * 100) if prior_total else 0:+.1f}%",
                len(variance_df),
                datetime.now().strftime("%d.%m.%Y %H:%M"),
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Übersicht', index=False)

        # Sheet 2: Varianzdetails (Variance Details)
        variance_export = variance_df.copy()
        variance_export = variance_export.rename(columns={
            'account': 'Konto',
            'account_name': 'Bezeichnung',
            'prior': 'Vorjahr',
            'current': 'Aktuell',
            'delta': 'Abweichung',
            'delta_pct': 'Abw. %',
            'abs_delta': '|Abweichung|',
            'share_of_total_abs_delta': 'Anteil',
        })
        # Format percentage
        if 'Abw. %' in variance_export.columns:
            variance_export['Abw. %'] = variance_export['Abw. %'].apply(
                lambda x: f"{x:.1%}" if pd.notna(x) else ""
            )
        if 'Anteil' in variance_export.columns:
            variance_export['Anteil'] = variance_export['Anteil'].apply(
                lambda x: f"{x:.1%}" if pd.notna(x) else ""
            )
        variance_export.to_excel(writer, sheet_name='Varianzdetails', index=False)

        # Sheet 3: Kostenstellen (if provided)
        if cost_center_summary is not None and not cost_center_summary.empty:
            cc_export = cost_center_summary.copy()
            cc_export = cc_export.rename(columns={
                'cost_center': 'Kostenstelle',
                'prior': 'Vorjahr',
                'current': 'Aktuell',
                'delta': 'Abweichung',
                'delta_pct': 'Abw. %',
                'abs_delta': '|Abweichung|',
                'share': 'Anteil',
            })
            if 'Abw. %' in cc_export.columns:
                cc_export['Abw. %'] = cc_export['Abw. %'].apply(
                    lambda x: f"{x:.1%}" if pd.notna(x) else ""
                )
            if 'Anteil' in cc_export.columns:
                cc_export['Anteil'] = cc_export['Anteil'].apply(
                    lambda x: f"{x:.1%}" if pd.notna(x) else ""
                )
            cc_export.to_excel(writer, sheet_name='Kostenstellen', index=False)

        # Sheet 4: Executive Summary (if provided)
        if executive_summary:
            exec_rows = []

            if executive_summary.get('headline'):
                exec_rows.append({'Abschnitt': 'Headline', 'Inhalt': executive_summary['headline']})
                exec_rows.append({'Abschnitt': '', 'Inhalt': ''})

            if executive_summary.get('key_findings'):
                exec_rows.append({'Abschnitt': 'Wichtigste Erkenntnisse', 'Inhalt': ''})
                for finding in executive_summary['key_findings']:
                    if finding and isinstance(finding, str):
                        exec_rows.append({'Abschnitt': '', 'Inhalt': f"• {finding}"})
                exec_rows.append({'Abschnitt': '', 'Inhalt': ''})

            if executive_summary.get('top_variances'):
                exec_rows.append({'Abschnitt': 'Top Abweichungen', 'Inhalt': ''})
                for v in executive_summary['top_variances']:
                    if isinstance(v, dict):
                        name = v.get('name', '')
                        delta = v.get('delta', 0)
                        reason = v.get('reason', '')
                        if name:
                            exec_rows.append({
                                'Abschnitt': '',
                                'Inhalt': f"• {name}: {delta:+,.0f} EUR - {reason}"
                            })
                exec_rows.append({'Abschnitt': '', 'Inhalt': ''})

            if executive_summary.get('patterns'):
                exec_rows.append({'Abschnitt': 'Erkannte Muster', 'Inhalt': ''})
                for pattern in executive_summary['patterns']:
                    if pattern and isinstance(pattern, str):
                        exec_rows.append({'Abschnitt': '', 'Inhalt': f"• {pattern}"})
                exec_rows.append({'Abschnitt': '', 'Inhalt': ''})

            if executive_summary.get('recommendations'):
                exec_rows.append({'Abschnitt': 'Empfehlungen', 'Inhalt': ''})
                for rec in executive_summary['recommendations']:
                    if rec and isinstance(rec, str):
                        exec_rows.append({'Abschnitt': '', 'Inhalt': f"• {rec}"})
                exec_rows.append({'Abschnitt': '', 'Inhalt': ''})

            if executive_summary.get('open_items'):
                exec_rows.append({'Abschnitt': 'Klärungsbedarf', 'Inhalt': ''})
                for item in executive_summary['open_items']:
                    if item and isinstance(item, str):
                        exec_rows.append({'Abschnitt': '', 'Inhalt': f"• {item}"})

            if exec_rows:
                exec_df = pd.DataFrame(exec_rows)
                exec_df.to_excel(writer, sheet_name='Executive Summary', index=False)

    output.seek(0)
    return output.getvalue()


def generate_cost_center_summary(
    prior_df: pd.DataFrame,
    curr_df: pd.DataFrame,
) -> pd.DataFrame:
    """Generate variance summary by cost center.

    Args:
        prior_df: Prior period postings
        curr_df: Current period postings

    Returns:
        DataFrame with variance by cost center
    """
    # Aggregate by cost center
    prior_cc = prior_df.groupby('cost_center')['amount'].sum().reset_index()
    prior_cc.columns = ['cost_center', 'prior']

    curr_cc = curr_df.groupby('cost_center')['amount'].sum().reset_index()
    curr_cc.columns = ['cost_center', 'current']

    # Merge
    merged = pd.merge(prior_cc, curr_cc, on='cost_center', how='outer').fillna(0)

    # Calculate metrics
    merged['delta'] = merged['current'] - merged['prior']
    merged['delta_pct'] = merged.apply(
        lambda r: r['delta'] / r['prior'] if r['prior'] != 0 else None, axis=1
    )
    merged['abs_delta'] = merged['delta'].abs()

    total_abs = merged['abs_delta'].sum()
    merged['share'] = merged['abs_delta'] / total_abs if total_abs > 0 else 0

    # Sort by absolute delta
    merged = merged.sort_values('abs_delta', ascending=False)

    return merged
