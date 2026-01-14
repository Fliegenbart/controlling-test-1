"""PDF Report Generator for Variance Copilot."""

from __future__ import annotations

from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional

from fpdf import FPDF


def _sanitize_text(text: str) -> str:
    """Sanitize text for PDF rendering with standard fonts.

    Replaces Unicode characters that aren't supported by Helvetica
    with ASCII equivalents.
    """
    if not text:
        return text

    # Common replacements for characters outside latin-1
    replacements = {
        '\u2013': '-',   # en-dash
        '\u2014': '-',   # em-dash
        '\u2018': "'",   # left single quote
        '\u2019': "'",   # right single quote
        '\u201c': '"',   # left double quote
        '\u201d': '"',   # right double quote
        '\u2026': '...', # ellipsis
        '\u2022': '*',   # bullet
        '\u00b7': '*',   # middle dot
        '\u2212': '-',   # minus sign
        '\u00a0': ' ',   # non-breaking space
        '\u20ac': 'EUR', # euro sign
    }

    for char, replacement in replacements.items():
        text = text.replace(char, replacement)

    # Remove any remaining non-latin-1 characters
    try:
        text.encode('latin-1')
    except UnicodeEncodeError:
        # Filter out characters that can't be encoded
        text = ''.join(c if ord(c) < 256 else '?' for c in text)

    return text


class VarianceReportPDF(FPDF):
    """Custom PDF class for variance reports."""

    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=20)

        # Colors (RGB)
        self.color_primary = (29, 29, 31)  # Dark text
        self.color_secondary = (134, 134, 139)  # Gray text
        self.color_accent = (0, 113, 227)  # Blue accent
        self.color_success = (36, 138, 61)  # Green
        self.color_warning = (178, 80, 0)  # Orange
        self.color_light_bg = (245, 245, 247)  # Light background

    def header(self):
        """Page header."""
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*self.color_secondary)
        self.cell(0, 10, "Clarity Report", align="L")
        self.cell(0, 10, datetime.now().strftime("%d.%m.%Y"), align="R", new_x="LMARGIN", new_y="NEXT")
        self.ln(5)

    def footer(self):
        """Page footer."""
        self.set_y(-15)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*self.color_secondary)
        self.cell(0, 10, f"Seite {self.page_no()}/{{nb}}", align="C")

    def add_title(self, title: str):
        """Add main report title."""
        self.set_font("Helvetica", "B", 24)
        self.set_text_color(*self.color_primary)
        self.cell(0, 15, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(5)

    def add_subtitle(self, subtitle: str):
        """Add subtitle/description."""
        self.set_font("Helvetica", "", 12)
        self.set_text_color(*self.color_secondary)
        self.multi_cell(0, 6, _sanitize_text(subtitle))
        self.ln(5)

    def add_section_header(self, title: str):
        """Add section header."""
        self.ln(5)
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(*self.color_primary)
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def add_metric_row(self, metrics: List[Dict[str, str]]):
        """Add a row of metrics."""
        col_width = (self.w - 20) / len(metrics)
        self.set_fill_color(*self.color_light_bg)

        for metric in metrics:
            x_start = self.get_x()
            y_start = self.get_y()

            # Background
            self.rect(x_start, y_start, col_width - 5, 25, style="F")

            # Label
            self.set_xy(x_start + 5, y_start + 3)
            self.set_font("Helvetica", "", 8)
            self.set_text_color(*self.color_secondary)
            self.cell(col_width - 10, 5, metric.get("label", ""), new_x="LEFT", new_y="NEXT")

            # Value
            self.set_x(x_start + 5)
            self.set_font("Helvetica", "B", 14)
            self.set_text_color(*self.color_primary)
            self.cell(col_width - 10, 8, metric.get("value", ""))

            self.set_xy(x_start + col_width, y_start)

        self.ln(30)
        self.set_x(self.l_margin)

    def add_variance_table(self, data: List[Dict[str, Any]], headers: List[str]):
        """Add variance overview table."""
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(*self.color_light_bg)
        self.set_text_color(*self.color_primary)

        # Calculate column widths
        col_widths = [25, 50, 25, 25, 25, 20, 20]
        total_width = sum(col_widths)
        scale = (self.w - 20) / total_width
        col_widths = [w * scale for w in col_widths]

        # Header row
        for i, header in enumerate(headers):
            self.cell(col_widths[i], 8, header, border=1, fill=True, align="C")
        self.ln()

        # Data rows
        self.set_font("Helvetica", "", 8)
        for row in data:
            values = [
                str(row.get("account", "")),
                str(row.get("account_name", ""))[:25],
                str(row.get("prior", "")),
                str(row.get("current", "")),
                str(row.get("delta", "")),
                str(row.get("delta_pct", "")),
                str(row.get("share", "")),
            ]
            for i, val in enumerate(values):
                align = "L" if i == 1 else "R"
                self.cell(col_widths[i], 7, val, border=1, align=align)
            self.ln()

    def add_account_analysis(
        self,
        account: str,
        account_name: str,
        prior: float,
        current: float,
        delta: float,
        delta_pct: Optional[float],
        analysis: Optional[Dict[str, Any]],
    ):
        """Add single account analysis section."""
        # Account header
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(*self.color_accent)
        self.cell(0, 8, f"{account} - {account_name}", new_x="LMARGIN", new_y="NEXT")

        # Metrics
        pct_str = f"{delta_pct:+.1%}" if delta_pct is not None else "-"
        self.add_metric_row([
            {"label": "VORJAHR", "value": f"{prior:,.0f}"},
            {"label": "AKTUELL", "value": f"{current:,.0f}"},
            {"label": "VARIANZ", "value": f"{delta:+,.0f} ({pct_str})"},
        ])

        if analysis:
            # Headline
            if analysis.get("headline"):
                self.set_font("Helvetica", "B", 11)
                self.set_text_color(*self.color_primary)
                self.multi_cell(0, 6, _sanitize_text(analysis["headline"]))
                self.ln(3)

            # Summary bullets
            if analysis.get("summary"):
                self.set_font("Helvetica", "", 10)
                self.set_text_color(*self.color_primary)
                for bullet in analysis["summary"]:
                    if not bullet:
                        continue
                    # Clean up evidence labels for display
                    bullet_clean = bullet.replace("[Datenbasiert]", "[D]").replace("[Indiz]", "[I]").replace("[Offen]", "[O]")
                    self.set_x(self.l_margin)
                    self.multi_cell(0, 5, _sanitize_text(f"  {bullet_clean}"))
                self.ln(3)

            # Evidence section
            if analysis.get("evidence"):
                self.set_font("Helvetica", "B", 10)
                self.set_text_color(*self.color_primary)
                self.cell(0, 6, "Evidenz:", new_x="LMARGIN", new_y="NEXT")

                self.set_font("Helvetica", "", 9)
                for ev in analysis["evidence"]:
                    label = ev.get("label", "")
                    text = ev.get("text", "")
                    if not text:
                        continue

                    # Color based on label
                    if label == "Datenbasiert":
                        self.set_text_color(*self.color_success)
                        prefix = "[D]"
                    elif label == "Indiz":
                        self.set_text_color(*self.color_warning)
                        prefix = "[I]"
                    else:
                        self.set_text_color(*self.color_accent)
                        prefix = "[O]"

                    self.set_x(self.l_margin)
                    self.set_font("Helvetica", "B", 9)
                    self.cell(10, 5, prefix)
                    self.set_font("Helvetica", "", 9)
                    self.set_text_color(*self.color_primary)
                    self.multi_cell(0, 5, _sanitize_text(text))
                self.ln(2)

            # Open questions
            if analysis.get("questions"):
                self.set_font("Helvetica", "B", 10)
                self.set_text_color(*self.color_primary)
                self.cell(0, 6, "Offene Fragen:", new_x="LMARGIN", new_y="NEXT")

                self.set_font("Helvetica", "", 9)
                self.set_text_color(*self.color_secondary)
                for q in analysis["questions"]:
                    if not q:
                        continue
                    self.set_x(self.l_margin)
                    self.multi_cell(0, 5, _sanitize_text(f"  ? {q}"))
                self.ln(2)

        self.ln(5)

    def add_executive_summary(self, summary: Dict[str, Any]):
        """Add executive summary section."""
        self.add_section_header("Executive Summary")

        if summary.get("headline"):
            self.set_font("Helvetica", "B", 12)
            self.set_text_color(*self.color_primary)
            self.multi_cell(0, 7, _sanitize_text(summary["headline"]))
            self.ln(5)

        if summary.get("key_findings"):
            self.set_font("Helvetica", "B", 10)
            self.set_text_color(*self.color_primary)
            self.cell(0, 6, "Wichtigste Erkenntnisse:", new_x="LMARGIN", new_y="NEXT")

            self.set_font("Helvetica", "", 10)
            for finding in summary["key_findings"]:
                if not finding:
                    continue
                self.set_x(self.l_margin)
                self.multi_cell(0, 5, _sanitize_text(f"  {finding}"))
            self.ln(3)

        if summary.get("top_variances"):
            self.set_font("Helvetica", "B", 10)
            self.cell(0, 6, "Top Abweichungen:", new_x="LMARGIN", new_y="NEXT")

            self.set_font("Helvetica", "", 10)
            for var in summary["top_variances"]:
                name = var.get("name", "")
                if not name:
                    continue
                delta = var.get("delta", 0)
                reason = var.get("reason", "")
                self.set_x(self.l_margin)
                self.multi_cell(0, 5, _sanitize_text(f"  {name}: {delta:+,.0f} EUR - {reason}"))
            self.ln(3)

        if summary.get("recommendations"):
            self.set_font("Helvetica", "B", 10)
            self.cell(0, 6, "Empfehlungen:", new_x="LMARGIN", new_y="NEXT")

            self.set_font("Helvetica", "", 10)
            for rec in summary["recommendations"]:
                if not rec:
                    continue
                self.set_x(self.l_margin)
                self.multi_cell(0, 5, _sanitize_text(f"  {rec}"))
            self.ln(3)

        if summary.get("open_items"):
            self.set_font("Helvetica", "B", 10)
            self.set_text_color(*self.color_warning)
            self.cell(0, 6, "Klärungsbedarf:", new_x="LMARGIN", new_y="NEXT")

            self.set_font("Helvetica", "", 10)
            self.set_text_color(*self.color_primary)
            for item in summary["open_items"]:
                if not item:
                    continue
                self.set_x(self.l_margin)
                self.multi_cell(0, 5, _sanitize_text(f"  ? {item}"))


def generate_single_account_pdf(
    account: str,
    account_name: str,
    prior: float,
    current: float,
    delta: float,
    delta_pct: Optional[float],
    analysis: Optional[Dict[str, Any]],
    drivers: Optional[List[Dict[str, Any]]] = None,
) -> bytes:
    """Generate PDF for single account analysis.

    Args:
        account: Account number
        account_name: Account description
        prior: Prior period value
        current: Current period value
        delta: Variance
        delta_pct: Variance percentage
        analysis: AI analysis result dict
        drivers: Optional list of drivers

    Returns:
        PDF as bytes
    """
    pdf = VarianceReportPDF()
    pdf.alias_nb_pages()
    pdf.add_page()

    # Title
    pdf.add_title("Varianzanalyse")
    pdf.add_subtitle(f"Konto {account} - {account_name}")

    # Account analysis
    pdf.add_account_analysis(
        account=account,
        account_name=account_name,
        prior=prior,
        current=current,
        delta=delta,
        delta_pct=delta_pct,
        analysis=analysis,
    )

    # Drivers table if provided
    if drivers:
        pdf.add_section_header("Treiber-Analyse")
        pdf.set_font("Helvetica", "", 9)
        for d in drivers[:10]:
            name = d.get("cost_center") or d.get("vendor") or "-"
            d_delta = d.get("delta", 0)
            share = d.get("share", 0)
            pdf.cell(0, 5, f"  {name}: {d_delta:+,.0f} ({share:.0%})", new_x="LMARGIN", new_y="NEXT")

    # Footer info
    pdf.ln(10)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(*pdf.color_secondary)
    pdf.cell(0, 5, "Generiert mit Clarity - Alle Daten lokal verarbeitet", align="C")

    return bytes(pdf.output())


def generate_executive_summary_pdf(
    title: str,
    period_info: str,
    total_prior: float,
    total_current: float,
    total_delta: float,
    variance_data: List[Dict[str, Any]],
    executive_summary: Dict[str, Any],
    account_analyses: Optional[List[Dict[str, Any]]] = None,
) -> bytes:
    """Generate comprehensive executive summary PDF.

    Args:
        title: Report title
        period_info: Period description (e.g., "Q2 2024 vs Q2 2025")
        total_prior: Total prior period value
        total_current: Total current period value
        total_delta: Total variance
        variance_data: List of variance rows for table
        executive_summary: AI-generated executive summary
        account_analyses: Optional list of individual account analyses

    Returns:
        PDF as bytes
    """
    pdf = VarianceReportPDF()
    pdf.alias_nb_pages()
    pdf.add_page()

    # Title page
    pdf.add_title(title)
    pdf.add_subtitle(f"Berichtszeitraum: {period_info}")
    pdf.ln(5)

    # Overview metrics
    pdf.add_section_header("Gesamtübersicht")
    delta_pct = (total_delta / total_prior * 100) if total_prior != 0 else 0
    pdf.add_metric_row([
        {"label": "VORJAHR GESAMT", "value": f"{total_prior:,.0f} EUR"},
        {"label": "AKTUELL GESAMT", "value": f"{total_current:,.0f} EUR"},
        {"label": "GESAMTVARIANZ", "value": f"{total_delta:+,.0f} EUR ({delta_pct:+.1f}%)"},
    ])

    # Executive summary
    if executive_summary:
        pdf.add_executive_summary(executive_summary)

    # Variance table
    if variance_data:
        pdf.add_page()
        pdf.add_section_header("Varianzübersicht")
        headers = ["Konto", "Bezeichnung", "VJ", "AQ", "Delta", "Delta%", "Anteil"]
        pdf.add_variance_table(variance_data[:20], headers)

    # Individual account analyses
    if account_analyses:
        pdf.add_page()
        pdf.add_section_header("Detailanalysen")

        for acc in account_analyses:
            if pdf.get_y() > 220:  # Check if we need a new page
                pdf.add_page()

            pdf.add_account_analysis(
                account=acc.get("account", ""),
                account_name=acc.get("account_name", ""),
                prior=acc.get("prior", 0),
                current=acc.get("current", 0),
                delta=acc.get("delta", 0),
                delta_pct=acc.get("delta_pct"),
                analysis=acc.get("analysis"),
            )

    # Footer
    pdf.ln(10)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(*pdf.color_secondary)
    pdf.cell(0, 5, "Generiert mit Clarity - 100% lokale Datenverarbeitung", align="C")

    return bytes(pdf.output())
