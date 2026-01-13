"""Variance Copilot - Streamlit App."""

from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from variance_copilot.io import load_csv
from variance_copilot.normalize import ColumnMapping, SignMode, normalize
from variance_copilot.variance import (
    drivers_for_account,
    materiality_filter,
    samples_for_account,
    variance_by_account,
)
from variance_copilot.keywords import keywords_for_account
from variance_copilot.ollama_client import (
    extract_json,
    get_config,
    is_available,
    list_models,
    ollama_generate,
)
from variance_copilot.prompts import SYSTEM_PROMPT, format_context

st.set_page_config(page_title="Variance Copilot", page_icon="📊", layout="wide")

# --- Session State ---
if "prior_df" not in st.session_state:
    st.session_state.prior_df = None
if "curr_df" not in st.session_state:
    st.session_state.curr_df = None
if "variance_df" not in st.session_state:
    st.session_state.variance_df = None
if "selected_account" not in st.session_state:
    st.session_state.selected_account = None
if "comment_result" not in st.session_state:
    st.session_state.comment_result = None

# --- Sidebar ---
st.sidebar.title("Einstellungen")
st.sidebar.warning("Keine echten Daten ins Repo!")

st.sidebar.subheader("Materialität")
min_abs_delta = st.sidebar.number_input("Min. Abs. Delta (EUR)", 0, 1000000, 5000, 1000)
min_pct_delta = st.sidebar.number_input("Min. % Delta", 0, 100, 10, 5) / 100
min_base = st.sidebar.number_input("Min. Basis (EUR)", 0, 1000000, 10000, 1000)

st.sidebar.subheader("Vorzeichen")
sign_mode = st.sidebar.selectbox(
    "Amount Sign",
    options=[SignMode.AS_IS, SignMode.INVERT, SignMode.ABS],
    format_func=lambda x: {"as_is": "Wie Export", "invert": "Invertieren", "abs": "Absolutwert"}[x.value],
)

st.sidebar.subheader("Treiber-Dimension")
dimension = st.sidebar.selectbox("Dimension", ["cost_center", "vendor"], format_func=lambda x: "Kostenstelle" if x == "cost_center" else "Lieferant")

st.sidebar.subheader("Ollama")
ollama_config = get_config()
ollama_url = st.sidebar.text_input("Base URL", ollama_config["base_url"])
ollama_model = st.sidebar.text_input("Model", ollama_config["model"])

if is_available(ollama_url):
    st.sidebar.success("Ollama erreichbar")
    models = list_models(ollama_url)
    if models:
        st.sidebar.caption(f"Modelle: {', '.join(models[:5])}")
else:
    st.sidebar.error("Ollama nicht erreichbar")

# --- Main ---
st.title("Variance Copilot")
st.caption("Quartals-Abweichungsanalyse YoY mit KI-Kommentierung (100% lokal)")

# --- Upload Section ---
st.header("1. Daten-Import")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Vorjahr")
    prior_file = st.file_uploader("CSV", type=["csv"], key="prior")

with col2:
    st.subheader("Aktuelles Quartal")
    curr_file = st.file_uploader("CSV", type=["csv"], key="curr")

if prior_file and curr_file:
    raw_prior = load_csv(BytesIO(prior_file.read()))
    raw_curr = load_csv(BytesIO(curr_file.read()))

    st.success(f"Geladen: Vorjahr {len(raw_prior)} Zeilen, Aktuell {len(raw_curr)} Zeilen")

    # --- Mapping ---
    st.subheader("Spalten-Mapping")
    cols = [""] + list(raw_prior.columns)

    c1, c2, c3 = st.columns(3)
    with c1:
        map_date = st.selectbox("Datum", cols, index=cols.index("posting_date") if "posting_date" in cols else 0)
        map_amount = st.selectbox("Betrag", cols, index=cols.index("amount") if "amount" in cols else 0)
    with c2:
        map_account = st.selectbox("Konto", cols, index=cols.index("account") if "account" in cols else 0)
        map_name = st.selectbox("Kontoname", cols, index=cols.index("account_name") if "account_name" in cols else 0)
    with c3:
        map_cc = st.selectbox("Kostenstelle", cols, index=cols.index("cost_center") if "cost_center" in cols else 0)
        map_vendor = st.selectbox("Lieferant", cols, index=cols.index("vendor") if "vendor" in cols else 0)
        map_text = st.selectbox("Buchungstext", cols, index=cols.index("text") if "text" in cols else 0)

    if st.button("Daten laden & validieren", type="primary"):
        if not map_date or not map_amount or not map_account:
            st.error("Pflichtfelder: Datum, Betrag, Konto")
        else:
            mapping = ColumnMapping(
                posting_date=map_date,
                amount=map_amount,
                account=map_account,
                account_name=map_name or None,
                cost_center=map_cc or None,
                vendor=map_vendor or None,
                text=map_text or None,
            )

            st.session_state.prior_df = normalize(raw_prior, mapping, sign_mode)
            st.session_state.curr_df = normalize(raw_curr, mapping, sign_mode)

            st.session_state.variance_df = variance_by_account(
                st.session_state.prior_df, st.session_state.curr_df
            )

            c1, c2 = st.columns(2)
            with c1:
                st.metric("Vorjahr Summe", f"{st.session_state.prior_df['amount'].sum():,.0f} EUR")
            with c2:
                st.metric("Aktuell Summe", f"{st.session_state.curr_df['amount'].sum():,.0f} EUR")

            st.success("Daten validiert!")

# --- Variance Overview ---
if st.session_state.variance_df is not None:
    st.header("2. Abweichungsübersicht")

    filtered = materiality_filter(
        st.session_state.variance_df,
        min_abs_delta=min_abs_delta if min_abs_delta > 0 else None,
        min_pct_delta=min_pct_delta if min_pct_delta > 0 else None,
        min_base=min_base if min_base > 0 else None,
    )

    st.caption(f"{len(filtered)} von {len(st.session_state.variance_df)} Konten nach Materialitätsfilter")

    display = filtered.copy()
    display["prior"] = display["prior"].apply(lambda x: f"{x:,.0f}")
    display["current"] = display["current"].apply(lambda x: f"{x:,.0f}")
    display["delta"] = display["delta"].apply(lambda x: f"{x:+,.0f}")
    display["delta_pct"] = display["delta_pct"].apply(lambda x: f"{x:+.1%}" if pd.notna(x) else "n/a")

    st.dataframe(
        display[["account", "account_name", "prior", "current", "delta", "delta_pct"]].rename(
            columns={"account": "Konto", "account_name": "Name", "prior": "VJ", "current": "AQ", "delta": "Delta", "delta_pct": "%"}
        ),
        use_container_width=True,
        hide_index=True,
    )

    # --- Drilldown ---
    st.header("3. Drill-Down")

    accounts = filtered["account"].tolist()
    if accounts:
        selected = st.selectbox(
            "Konto auswählen",
            accounts,
            format_func=lambda x: f"{x} - {filtered[filtered['account'] == x]['account_name'].iloc[0]}",
        )
        st.session_state.selected_account = selected

        acc_row = filtered[filtered["account"] == selected].iloc[0]

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Vorjahr", f"{acc_row['prior']}")
        with c2:
            st.metric("Aktuell", f"{acc_row['current']}")
        with c3:
            st.metric("Delta", f"{acc_row['delta']}", delta=f"{acc_row['delta_pct']}")

        # Drivers
        st.subheader("Treiber")
        drivers = drivers_for_account(
            st.session_state.prior_df, st.session_state.curr_df, selected, dimension
        )
        if not drivers.empty:
            drv_display = drivers.copy()
            drv_display["prior"] = drv_display["prior"].apply(lambda x: f"{x:,.0f}")
            drv_display["current"] = drv_display["current"].apply(lambda x: f"{x:,.0f}")
            drv_display["delta"] = drv_display["delta"].apply(lambda x: f"{x:+,.0f}")
            drv_display["share"] = drv_display["share"].apply(lambda x: f"{x:.0%}")
            st.dataframe(drv_display, use_container_width=True, hide_index=True)
        else:
            st.info("Keine Treiber-Daten")

        # Samples
        st.subheader("Top Buchungen")
        samples = samples_for_account(st.session_state.curr_df, selected)
        if not samples.empty:
            st.dataframe(samples, use_container_width=True, hide_index=True)
        else:
            st.info("Keine Buchungen")

        # Keywords
        st.subheader("Keywords")
        kw = keywords_for_account(st.session_state.curr_df, selected)
        if kw:
            st.write(", ".join(f"**{k}** ({c})" for k, c in kw[:10]))
        else:
            st.info("Keine Keywords")

        # --- Ollama Comment ---
        st.header("4. KI-Kommentar")

        if st.button("Kommentar generieren (Ollama)", type="primary"):
            if not is_available(ollama_url):
                st.error("Ollama nicht erreichbar. Bitte starten: `ollama serve`")
            else:
                with st.spinner("Generiere..."):
                    # Build context
                    drivers_list = drivers.to_dict("records") if not drivers.empty else []
                    samples_list = samples.to_dict("records") if not samples.empty else []

                    prior_val = st.session_state.variance_df[st.session_state.variance_df["account"] == selected]["prior"].iloc[0]
                    curr_val = st.session_state.variance_df[st.session_state.variance_df["account"] == selected]["current"].iloc[0]
                    delta_val = st.session_state.variance_df[st.session_state.variance_df["account"] == selected]["delta"].iloc[0]
                    pct_val = st.session_state.variance_df[st.session_state.variance_df["account"] == selected]["delta_pct"].iloc[0]

                    prompt = format_context(
                        account=selected,
                        account_name=acc_row["account_name"] or "",
                        prior=prior_val,
                        current=curr_val,
                        delta=delta_val,
                        delta_pct=pct_val,
                        drivers=drivers_list,
                        samples=samples_list,
                        keywords=kw,
                    )

                    try:
                        response = ollama_generate(
                            user_prompt=prompt,
                            system_prompt=SYSTEM_PROMPT,
                            base_url=ollama_url,
                            model=ollama_model,
                        )
                        st.session_state.comment_result = {
                            "raw": response,
                            "parsed": extract_json(response),
                        }
                    except (ConnectionError, RuntimeError) as e:
                        st.error(str(e))
                        st.session_state.comment_result = None

        if st.session_state.comment_result:
            result = st.session_state.comment_result

            if result["parsed"]:
                data = result["parsed"]
                st.success("Kommentar generiert!")

                st.markdown(f"### {data.get('headline', 'Kommentar')}")

                st.markdown("**Summary:**")
                for b in data.get("summary", []):
                    st.markdown(f"- {b}")

                if data.get("drivers"):
                    st.markdown("**Treiber:**")
                    for d in data["drivers"]:
                        st.markdown(f"- {d.get('name', 'n/a')}: {d.get('delta', 0):+,.0f} ({d.get('share', 0):.0%})")

                if data.get("evidence"):
                    st.markdown("**Evidenz:**")
                    for e in data["evidence"]:
                        label = e.get("label", "Offen")
                        color = {"Datenbasiert": "green", "Indiz": "orange"}.get(label, "red")
                        st.markdown(f"- :{color}[{label}]: {e.get('text', '')}")

                if data.get("questions"):
                    st.markdown("**Offene Fragen:**")
                    for q in data["questions"]:
                        st.markdown(f"- {q}")

                # Export
                st.divider()
                md = f"# {data.get('headline', 'Kommentar')}\n\n"
                md += "## Summary\n" + "\n".join(f"- {b}" for b in data.get("summary", [])) + "\n\n"
                md += "## Treiber\n" + "\n".join(
                    f"- {d.get('name')}: {d.get('delta', 0):+,.0f}" for d in data.get("drivers", [])
                ) + "\n"

                st.download_button("Download Markdown", md, f"comment_{selected}.md", "text/markdown")

                with st.expander("Raw JSON"):
                    st.json(data)
            else:
                st.warning("JSON konnte nicht geparst werden")
                with st.expander("Raw Response"):
                    st.text(result["raw"])
