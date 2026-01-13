"""Variance Copilot - Streamlit App with Apple-inspired Design."""

from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

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

# --- Page Config ---
st.set_page_config(
    page_title="Variance Copilot",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Apple-inspired CSS ---
st.markdown("""
<style>
    /* Import SF Pro-like font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global styles */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background: linear-gradient(180deg, #fafafa 0%, #f5f5f7 100%);
    }

    /* Header styling */
    h1 {
        font-weight: 600 !important;
        letter-spacing: -0.5px !important;
        color: #1d1d1f !important;
    }

    h2, h3 {
        font-weight: 500 !important;
        color: #1d1d1f !important;
    }

    /* Card-like containers */
    .stExpander, [data-testid="stExpander"] {
        background: white;
        border-radius: 12px;
        border: 1px solid #e5e5e5;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: white;
        padding: 20px;
        border-radius: 16px;
        border: 1px solid #e5e5e5;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }

    [data-testid="stMetricValue"] {
        font-size: 28px !important;
        font-weight: 600 !important;
        color: #1d1d1f !important;
    }

    [data-testid="stMetricLabel"] {
        font-size: 13px !important;
        font-weight: 500 !important;
        color: #86868b !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(180deg, #007AFF 0%, #0066DD 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: 500;
        font-size: 15px;
        transition: all 0.2s ease;
        box-shadow: 0 2px 8px rgba(0,122,255,0.3);
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,122,255,0.4);
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: white;
        border-radius: 12px;
        border: 2px dashed #d2d2d7;
        padding: 20px;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: #007AFF;
        background: #f5f9ff;
    }

    /* Dataframes */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }

    /* Select boxes */
    .stSelectbox > div > div {
        background: white;
        border-radius: 10px;
        border: 1px solid #d2d2d7;
    }

    /* Success/Error/Warning messages */
    .stSuccess, .stError, .stWarning, .stInfo {
        border-radius: 10px;
        border: none;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #f5f5f7;
        border-right: 1px solid #e5e5e5;
    }

    /* Divider */
    hr {
        border: none;
        height: 1px;
        background: #e5e5e5;
        margin: 24px 0;
    }

    /* Caption text */
    .stCaption {
        color: #86868b !important;
    }

    /* Number inputs */
    .stNumberInput > div > div > input {
        border-radius: 8px;
        border: 1px solid #d2d2d7;
    }

    /* Download button */
    .stDownloadButton > button {
        background: white;
        color: #007AFF;
        border: 1px solid #007AFF;
        border-radius: 10px;
    }

    .stDownloadButton > button:hover {
        background: #007AFF;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

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
with st.sidebar:
    st.markdown("### Settings")

    st.markdown("##### Materiality")
    min_abs_delta = st.number_input("Min. Absolute Delta", 0, 1000000, 5000, 1000)
    min_pct_delta = st.number_input("Min. % Delta", 0, 100, 10, 5) / 100
    min_base = st.number_input("Min. Base Value", 0, 1000000, 10000, 1000)

    st.markdown("##### Amount Sign")
    sign_mode = st.selectbox(
        "Mode",
        options=[SignMode.AS_IS, SignMode.INVERT, SignMode.ABS],
        format_func=lambda x: {"as_is": "As Exported", "invert": "Invert", "abs": "Absolute"}[x.value],
    )

    st.markdown("##### Driver Dimension")
    dimension = st.selectbox(
        "Group by",
        ["cost_center", "vendor"],
        format_func=lambda x: "Cost Center" if x == "cost_center" else "Vendor"
    )

    st.markdown("##### Ollama")
    ollama_config = get_config()
    ollama_url = st.text_input("Base URL", ollama_config["base_url"])
    ollama_model = st.text_input("Model", ollama_config["model"])

    if is_available(ollama_url):
        st.success("Connected", icon="✅")
    else:
        st.error("Not available", icon="❌")

# --- Main Content ---
st.markdown("# Variance Copilot")
st.caption("Quarterly YoY variance analysis with local AI commentary")

st.markdown("---")

# --- Upload Section ---
st.markdown("### Import Data")

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("**Prior Year (Same Quarter)**")
    prior_file = st.file_uploader("Upload CSV", type=["csv"], key="prior", label_visibility="collapsed")

with col2:
    st.markdown("**Current Quarter**")
    curr_file = st.file_uploader("Upload CSV", type=["csv"], key="curr", label_visibility="collapsed")

if prior_file and curr_file:
    raw_prior = load_csv(BytesIO(prior_file.read()))
    raw_curr = load_csv(BytesIO(curr_file.read()))

    st.success(f"Loaded: Prior {len(raw_prior):,} rows · Current {len(raw_curr):,} rows")

    st.markdown("---")
    st.markdown("### Column Mapping")

    cols = [""] + list(raw_prior.columns)

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        map_date = st.selectbox("Date", cols, index=cols.index("posting_date") if "posting_date" in cols else 0)
        map_amount = st.selectbox("Amount", cols, index=cols.index("amount") if "amount" in cols else 0)
    with c2:
        map_account = st.selectbox("Account", cols, index=cols.index("account") if "account" in cols else 0)
        map_name = st.selectbox("Account Name", cols, index=cols.index("account_name") if "account_name" in cols else 0)
    with c3:
        map_cc = st.selectbox("Cost Center", cols, index=cols.index("cost_center") if "cost_center" in cols else 0)
        map_vendor = st.selectbox("Vendor", cols, index=cols.index("vendor") if "vendor" in cols else 0)
        map_text = st.selectbox("Description", cols, index=cols.index("text") if "text" in cols else 0)

    st.markdown("")
    if st.button("Load & Validate", type="primary", use_container_width=False):
        if not map_date or not map_amount or not map_account:
            st.error("Required: Date, Amount, Account")
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

# --- Variance Overview ---
if st.session_state.variance_df is not None:
    st.markdown("---")
    st.markdown("### Overview")

    # Summary metrics
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        st.metric("Prior Year Total", f"{st.session_state.prior_df['amount'].sum():,.0f}")
    with c2:
        st.metric("Current Total", f"{st.session_state.curr_df['amount'].sum():,.0f}")
    with c3:
        total_delta = st.session_state.curr_df['amount'].sum() - st.session_state.prior_df['amount'].sum()
        st.metric("Total Variance", f"{total_delta:+,.0f}")
    with c4:
        st.metric("Accounts", f"{len(st.session_state.variance_df)}")

    st.markdown("")

    filtered = materiality_filter(
        st.session_state.variance_df,
        min_abs_delta=min_abs_delta if min_abs_delta > 0 else None,
        min_pct_delta=min_pct_delta if min_pct_delta > 0 else None,
        min_base=min_base if min_base > 0 else None,
    )

    st.caption(f"Showing {len(filtered)} of {len(st.session_state.variance_df)} accounts after materiality filter")

    display = filtered.copy()
    display["prior"] = display["prior"].apply(lambda x: f"{x:,.0f}")
    display["current"] = display["current"].apply(lambda x: f"{x:,.0f}")
    display["delta"] = display["delta"].apply(lambda x: f"{x:+,.0f}")
    display["delta_pct"] = display["delta_pct"].apply(lambda x: f"{x:+.1%}" if pd.notna(x) else "—")

    st.dataframe(
        display[["account", "account_name", "prior", "current", "delta", "delta_pct"]].rename(
            columns={"account": "Account", "account_name": "Name", "prior": "Prior", "current": "Current", "delta": "Δ", "delta_pct": "Δ%"}
        ),
        use_container_width=True,
        hide_index=True,
        height=300,
    )

    # --- Drill-Down ---
    st.markdown("---")
    st.markdown("### Analysis")

    accounts = filtered["account"].tolist()
    if accounts:
        selected = st.selectbox(
            "Select Account",
            accounts,
            format_func=lambda x: f"{x} — {filtered[filtered['account'] == x]['account_name'].iloc[0]}",
        )
        st.session_state.selected_account = selected

        acc_row = filtered[filtered["account"] == selected].iloc[0]
        prior_val = st.session_state.variance_df[st.session_state.variance_df["account"] == selected]["prior"].iloc[0]
        curr_val = st.session_state.variance_df[st.session_state.variance_df["account"] == selected]["current"].iloc[0]
        delta_val = st.session_state.variance_df[st.session_state.variance_df["account"] == selected]["delta"].iloc[0]
        pct_val = st.session_state.variance_df[st.session_state.variance_df["account"] == selected]["delta_pct"].iloc[0]

        st.markdown("")
        c1, c2, c3 = st.columns(3, gap="medium")
        with c1:
            st.metric("Prior Year", f"{prior_val:,.0f}")
        with c2:
            st.metric("Current", f"{curr_val:,.0f}")
        with c3:
            pct_str = f"{pct_val:+.1%}" if pd.notna(pct_val) else "—"
            st.metric("Variance", f"{delta_val:+,.0f}", delta=pct_str)

        st.markdown("")

        tab1, tab2, tab3 = st.tabs(["Drivers", "Top Postings", "Keywords"])

        with tab1:
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
                st.info("No driver data available")

        with tab2:
            samples = samples_for_account(st.session_state.curr_df, selected)
            if not samples.empty:
                st.dataframe(samples, use_container_width=True, hide_index=True)
            else:
                st.info("No postings found")

        with tab3:
            kw = keywords_for_account(st.session_state.curr_df, selected)
            if kw:
                kw_html = " ".join(
                    f'<span style="background:#f0f0f0;padding:4px 10px;border-radius:12px;margin:2px;display:inline-block;font-size:13px;">{k} <span style="color:#86868b;">({c})</span></span>'
                    for k, c in kw[:12]
                )
                st.markdown(kw_html, unsafe_allow_html=True)
            else:
                st.info("No keywords found")

        # --- AI Comment ---
        st.markdown("---")
        st.markdown("### AI Commentary")

        col1, col2 = st.columns([3, 1])
        with col1:
            generate_btn = st.button("Generate Comment", type="primary")
        with col2:
            if not is_available(ollama_url):
                st.caption("⚠️ Ollama offline")

        if generate_btn:
            if not is_available(ollama_url):
                st.error("Ollama is not available. Please run: `ollama serve`")
            else:
                with st.spinner("Generating..."):
                    drivers_list = drivers.to_dict("records") if not drivers.empty else []
                    samples_list = samples.to_dict("records") if not samples.empty else []

                    prompt = format_context(
                        account=selected,
                        account_name=acc_row["account_name"] or "",
                        prior=prior_val,
                        current=curr_val,
                        delta=delta_val,
                        delta_pct=pct_val,
                        drivers=drivers_list,
                        samples=samples_list,
                        keywords=kw if kw else [],
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

                st.markdown(f"#### {data.get('headline', 'Analysis')}")

                for b in data.get("summary", []):
                    st.markdown(f"• {b}")

                if data.get("drivers"):
                    st.markdown("**Key Drivers:**")
                    for d in data["drivers"]:
                        st.markdown(f"• {d.get('name', '—')}: {d.get('delta', 0):+,.0f} ({d.get('share', 0):.0%})")

                if data.get("evidence"):
                    st.markdown("**Evidence:**")
                    for e in data["evidence"]:
                        label = e.get("label", "Open")
                        icon = {"Datenbasiert": "✓", "Indiz": "○", "Offen": "?"}.get(label, "•")
                        st.markdown(f"{icon} **{label}:** {e.get('text', '')}")

                if data.get("questions"):
                    st.markdown("**Open Questions:**")
                    for q in data["questions"]:
                        st.markdown(f"• {q}")

                st.markdown("")

                md = f"# {data.get('headline', 'Comment')}\n\n"
                md += "## Summary\n" + "\n".join(f"- {b}" for b in data.get("summary", [])) + "\n\n"
                md += "## Drivers\n" + "\n".join(
                    f"- {d.get('name')}: {d.get('delta', 0):+,.0f}" for d in data.get("drivers", [])
                ) + "\n"

                st.download_button(
                    "Download as Markdown",
                    md,
                    f"comment_{selected}.md",
                    "text/markdown",
                )

                with st.expander("View JSON"):
                    st.json(data)
            else:
                st.warning("Could not parse JSON response")
                with st.expander("Raw Response"):
                    st.code(result["raw"])

# --- Footer ---
st.markdown("---")
st.caption("Variance Copilot • 100% Local • No data leaves your machine")
