#!/usr/bin/env python3
"""Generate synthetic sample data for Variance Copilot testing.

Convention: Revenue positive, Expenses negative.
"""

import random
from datetime import date, timedelta
from pathlib import Path

# Seed for reproducibility
random.seed(42)

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "sample_data"

# Account master data
ACCOUNTS = {
    # Revenue accounts (positive)
    "4000": ("Umsatzerlöse Labordiagnostik", "revenue"),
    "4010": ("Umsatzerlöse Beratung", "revenue"),
    "4020": ("Umsatzerlöse Schulungen", "revenue"),
    "4100": ("Sonstige Erlöse", "revenue"),
    # Expense accounts (negative)
    "5000": ("Material Reagenzien", "expense"),
    "5010": ("Material Verbrauchsmittel", "expense"),
    "5020": ("Fremdleistungen Labor", "expense"),
    "6000": ("Personalkosten Gehälter", "expense"),
    "6010": ("Personalkosten Sozialabgaben", "expense"),
    "6100": ("Miete Laborräume", "expense"),
    "6200": ("Wartung Geräte", "expense"),
    "6300": ("IT-Kosten", "expense"),
    "6400": ("Reisekosten", "expense"),
    "6500": ("Beratungskosten", "expense"),
    "6600": ("Versicherungen", "expense"),
    "6700": ("Marketing", "expense"),
    "6800": ("Fortbildung", "expense"),
    "6900": ("Sonstige Aufwendungen", "expense"),
}

# Cost centers
COST_CENTERS = ["CC100", "CC200", "CC300", "CC400", "CC500"]

# Profit centers
PROFIT_CENTERS = ["PC-DIAG", "PC-CONSULT", "PC-TRAIN", "PC-ADMIN"]

# Vendors (for expenses)
VENDORS = {
    "5000": ["Roche Diagnostics", "Siemens Healthineers", "Abbott", "Beckman Coulter"],
    "5010": ["VWR International", "Fisher Scientific", "Merck"],
    "5020": ["Synlab", "Bioscientia", "Sonic Healthcare"],
    "6000": ["Lohnbüro Müller"],
    "6100": ["Immobilien AG"],
    "6200": ["TechService GmbH", "Roche Diagnostics", "Siemens Healthineers"],
    "6300": ["SAP Deutschland", "Microsoft", "IT-Service Partner"],
    "6400": ["Lufthansa", "Deutsche Bahn", "Booking.com"],
    "6500": ["McKinsey", "Deloitte", "KPMG"],
    "6600": ["Allianz", "AXA"],
    "6700": ["Agentur Kreativ", "Google Ads"],
    "6800": ["Akademie Medizin", "Springer Verlag"],
    "6900": ["Diverse"],
}

# Customers (for revenue)
CUSTOMERS = [
    "Universitätsklinikum München",
    "Charité Berlin",
    "Uniklinik Köln",
    "Städtisches Klinikum Stuttgart",
    "MVZ Labor Bremen",
    "Praxis Dr. Weber",
    "Krankenhaus Nordwest",
    "Labor Gemeinschaft Bayern",
]

# Text templates
EXPENSE_TEXTS = {
    "5000": ["Reagenzien {month}", "Lieferung Testkits", "Nachbestellung Assays", "Chemikalien Q{q}"],
    "5010": ["Pipettenspitzen", "Handschuhe Nachbestellung", "Verbrauchsmaterial {month}"],
    "5020": ["Fremdanalyse Spezialtest", "Externe Laborleistung", "Auftragsanalyse"],
    "6000": ["Gehälter {month} {year}", "Lohnabrechnung {month}"],
    "6010": ["Sozialabgaben {month} {year}"],
    "6100": ["Miete {month} {year}", "Nebenkosten Q{q}"],
    "6200": ["Wartung Analysegerät", "Service Zentrifuge", "Inspektion Automat", "Große Revision Cobas"],
    "6300": ["SAP Lizenzen Q{q}", "Microsoft 365", "Serverkosten {month}"],
    "6400": ["Dienstreise {month}", "Kongress Teilnahme", "Kundenbesuch"],
    "6500": ["Strategieberatung", "Prozessoptimierung", "Gutachten"],
    "6600": ["Betriebshaftpflicht Q{q}", "Geräteversicherung"],
    "6700": ["Online Kampagne {month}", "Messebeteiligung", "Broschüren"],
    "6800": ["Schulung Mitarbeiter", "Fachkongress", "Zertifizierung"],
    "6900": ["Sonstiges {month}", "Kleinmaterial", "Bürobedarf"],
}

REVENUE_TEXTS = [
    "Diagnostik Auftrag {doc}",
    "Laboranalysen {month}",
    "Beratungsleistung",
    "Schulung Laborpersonal",
    "Projektabrechnung Q{q}",
    "Rahmenvertrag Abrechnung",
]


def random_date(year: int, quarter: int) -> date:
    """Generate random date in given quarter."""
    month_start = (quarter - 1) * 3 + 1
    month = random.randint(month_start, month_start + 2)
    day = random.randint(1, 28)
    return date(year, month, day)


def generate_doc_no(prefix: str, idx: int) -> str:
    """Generate document number."""
    return f"{prefix}{idx:06d}"


def generate_postings(year: int, quarter: int, is_2025: bool = False) -> list[dict]:
    """Generate postings for one quarter."""
    postings = []
    doc_idx = 100000

    months = {1: "Januar", 2: "Februar", 3: "März", 4: "April", 5: "Mai", 6: "Juni",
              7: "Juli", 8: "August", 9: "September", 10: "Oktober", 11: "November", 12: "Dezember"}

    # --- Revenue postings ---
    for _ in range(random.randint(120, 180)):
        account = random.choice(["4000", "4010", "4020", "4100"])
        account_name = ACCOUNTS[account][0]

        # Base amount
        if account == "4000":
            base = random.uniform(15000, 85000)
            # 2025 effect: more revenue
            if is_2025:
                base *= random.uniform(1.08, 1.18)
        elif account == "4010":
            base = random.uniform(8000, 35000)
        elif account == "4020":
            base = random.uniform(3000, 15000)
        else:
            base = random.uniform(1000, 8000)

        amount = round(base, 2)
        d = random_date(year, quarter)
        month_name = months[d.month]

        postings.append({
            "posting_date": d.isoformat(),
            "amount": amount,
            "account": account,
            "account_name": account_name,
            "cost_center": random.choice(COST_CENTERS[:3]),
            "profit_center": random.choice(PROFIT_CENTERS[:3]),
            "vendor": "",
            "customer": random.choice(CUSTOMERS),
            "document_no": generate_doc_no("RE", doc_idx),
            "text": random.choice(REVENUE_TEXTS).format(
                doc=doc_idx, month=month_name, q=quarter
            ),
        })
        doc_idx += 1

    # --- Expense postings ---
    for account, (account_name, _) in ACCOUNTS.items():
        if not account.startswith("4"):  # Expense accounts

            # Number of postings per account
            if account in ["6000", "6010", "6100"]:
                n_postings = 3  # Monthly
            elif account == "5000":
                n_postings = random.randint(45, 70)
            elif account in ["5010", "5020", "6200"]:
                n_postings = random.randint(15, 30)
            else:
                n_postings = random.randint(8, 20)

            for _ in range(n_postings):
                d = random_date(year, quarter)
                month_name = months[d.month]

                # Base amounts by account
                if account == "5000":
                    base = random.uniform(5000, 45000)
                    vendor = random.choice(VENDORS["5000"])
                    # 2025 effect: higher costs at Roche and Siemens
                    if is_2025 and vendor in ["Roche Diagnostics", "Siemens Healthineers"]:
                        base *= random.uniform(1.25, 1.45)
                elif account == "6000":
                    base = random.uniform(180000, 220000)
                    vendor = VENDORS["6000"][0]
                elif account == "6010":
                    base = random.uniform(35000, 45000)
                    vendor = VENDORS["6000"][0]
                elif account == "6100":
                    base = random.uniform(25000, 35000)
                    vendor = VENDORS["6100"][0]
                elif account == "6200":
                    base = random.uniform(3000, 15000)
                    vendor = random.choice(VENDORS.get(account, ["Diverse"]))
                else:
                    base = random.uniform(1000, 20000)
                    vendor = random.choice(VENDORS.get(account, ["Diverse"]))

                amount = -round(base, 2)  # Expenses negative

                text_templates = EXPENSE_TEXTS.get(account, ["Buchung {month}"])
                text = random.choice(text_templates).format(
                    month=month_name, year=year, q=quarter
                )

                postings.append({
                    "posting_date": d.isoformat(),
                    "amount": amount,
                    "account": account,
                    "account_name": account_name,
                    "cost_center": random.choice(COST_CENTERS),
                    "profit_center": random.choice(PROFIT_CENTERS),
                    "vendor": vendor,
                    "customer": "",
                    "document_no": generate_doc_no("BK", doc_idx),
                    "text": text,
                })
                doc_idx += 1

    # --- 2025 Special: One-off maintenance booking ---
    if is_2025:
        postings.append({
            "posting_date": date(year, 5, 15).isoformat(),
            "amount": -120000.00,
            "account": "6200",
            "account_name": "Wartung Geräte",
            "cost_center": "CC100",
            "profit_center": "PC-DIAG",
            "vendor": "Roche Diagnostics",
            "customer": "",
            "document_no": "BK999999",
            "text": "Große Revision Cobas 8000 - Sonderinspektion",
        })

    # Sort by date
    postings.sort(key=lambda x: x["posting_date"])
    return postings


def write_csv(postings: list[dict], filename: str):
    """Write postings to CSV file."""
    filepath = OUTPUT_DIR / filename

    columns = [
        "posting_date", "amount", "account", "account_name",
        "cost_center", "profit_center", "vendor", "customer",
        "document_no", "text"
    ]

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(",".join(columns) + "\n")
        for p in postings:
            row = [str(p.get(c, "")) for c in columns]
            # Escape commas in text fields
            row = [f'"{v}"' if "," in v else v for v in row]
            f.write(",".join(row) + "\n")

    print(f"Written {len(postings)} rows to {filepath}")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Generate Q2 2024
    postings_2024 = generate_postings(2024, 2, is_2025=False)
    write_csv(postings_2024, "buchungen_Q2_2024_fiktiv.csv")

    # Generate Q2 2025
    postings_2025 = generate_postings(2025, 2, is_2025=True)
    write_csv(postings_2025, "buchungen_Q2_2025_fiktiv.csv")

    print("\nSample data generated successfully!")
    print(f"  Q2 2024: {len(postings_2024)} postings")
    print(f"  Q2 2025: {len(postings_2025)} postings")


if __name__ == "__main__":
    main()
