"""Clarity - Streamlit App with Premium Apple-inspired Design."""

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
from variance_copilot.llm import (
    Backend,
    BackendStatus,
    detect_backend,
    generate as llm_generate,
    extract_json,
    get_backend_info,
)
from variance_copilot.ollama_client import get_config as get_ollama_config
from variance_copilot.prompts import (
    get_system_prompt,
    format_context,
    SYSTEM_PROMPT_EXECUTIVE,
    format_executive_context,
)
from variance_copilot.pdf_report import (
    generate_single_account_pdf,
    generate_executive_summary_pdf,
)

# --- Page Config ---
st.set_page_config(
    page_title="Clarity",
    page_icon="◎",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Premium Apple-inspired CSS ---
st.markdown("""
<style>
    /* === FONTS === */
    @import url('https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@300;400;500;600;700&family=SF+Pro+Text:wght@400;500;600&display=swap');

    /* Fallback to system fonts that mimic SF Pro */
    :root {
        --font-display: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Segoe UI', Roboto, sans-serif;
        --font-text: -apple-system, BlinkMacSystemFont, 'SF Pro Text', 'Segoe UI', Roboto, sans-serif;

        /* Apple Color Palette */
        --bg-primary: #ffffff;
        --bg-secondary: #f5f5f7;
        --bg-tertiary: #fafafa;
        --text-primary: #1d1d1f;
        --text-secondary: #86868b;
        --text-tertiary: #aeaeb2;
        --accent-blue: #0071e3;
        --accent-blue-hover: #0077ed;
        --border-light: rgba(0, 0, 0, 0.06);
        --border-medium: rgba(0, 0, 0, 0.1);
        --shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.04);
        --shadow-md: 0 4px 16px rgba(0, 0, 0, 0.08);
        --shadow-lg: 0 8px 32px rgba(0, 0, 0, 0.12);
        --radius-sm: 10px;
        --radius-md: 14px;
        --radius-lg: 20px;
        --radius-xl: 24px;
    }

    /* === GLOBAL RESET === */
    .stApp {
        font-family: var(--font-text);
        background: linear-gradient(180deg, #ffffff 0%, #f5f5f7 50%, #f0f0f2 100%);
        background-attachment: fixed;
    }

    /* Main container */
    .main .block-container {
        padding: 2rem 3rem 4rem 3rem;
        max-width: 1400px;
    }

    /* === TYPOGRAPHY === */
    h1 {
        font-family: var(--font-display) !important;
        font-size: 2.75rem !important;
        font-weight: 600 !important;
        letter-spacing: -0.03em !important;
        color: var(--text-primary) !important;
        margin-bottom: 0.25rem !important;
        background: linear-gradient(135deg, #1d1d1f 0%, #424245 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    h2 {
        font-family: var(--font-display) !important;
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em !important;
        color: var(--text-primary) !important;
        margin-top: 2rem !important;
    }

    h3 {
        font-family: var(--font-display) !important;
        font-size: 1.25rem !important;
        font-weight: 600 !important;
        letter-spacing: -0.01em !important;
        color: var(--text-primary) !important;
    }

    h4, h5, h6 {
        font-family: var(--font-text) !important;
        font-weight: 600 !important;
        color: var(--text-primary) !important;
    }

    p, span, div {
        font-family: var(--font-text);
        color: var(--text-primary);
        line-height: 1.5;
    }

    /* Caption/subtitle styling */
    .stCaption, [data-testid="stCaptionContainer"] {
        font-size: 1.05rem !important;
        color: var(--text-secondary) !important;
        font-weight: 400 !important;
        letter-spacing: 0 !important;
    }

    /* === GLASSMORPHISM CARDS === */
    .glass-card {
        background: rgba(255, 255, 255, 0.72);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: var(--radius-lg);
        border: 1px solid var(--border-light);
        box-shadow: var(--shadow-md);
        padding: 1.5rem;
    }

    /* === METRIC CARDS === */
    [data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: var(--radius-lg);
        border: 1px solid var(--border-light);
        box-shadow: var(--shadow-sm);
        padding: 1.25rem 1.5rem;
        transition: all 0.3s cubic-bezier(0.25, 0.1, 0.25, 1);
    }

    [data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-md);
        border-color: var(--border-medium);
    }

    [data-testid="stMetricLabel"] {
        font-family: var(--font-text) !important;
        font-size: 0.75rem !important;
        font-weight: 500 !important;
        color: var(--text-secondary) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
        margin-bottom: 0.5rem !important;
    }

    [data-testid="stMetricValue"] {
        font-family: var(--font-display) !important;
        font-size: 1.875rem !important;
        font-weight: 600 !important;
        color: var(--text-primary) !important;
        letter-spacing: -0.02em !important;
        line-height: 1.2 !important;
    }

    [data-testid="stMetricDelta"] {
        font-family: var(--font-text) !important;
        font-size: 0.875rem !important;
        font-weight: 500 !important;
    }

    /* === PRIMARY BUTTONS === */
    .stButton > button[kind="primary"],
    .stButton > button {
        font-family: var(--font-text) !important;
        background: var(--accent-blue) !important;
        color: white !important;
        border: none !important;
        border-radius: var(--radius-md) !important;
        padding: 0.75rem 1.5rem !important;
        font-size: 0.9375rem !important;
        font-weight: 500 !important;
        letter-spacing: -0.01em !important;
        transition: all 0.2s cubic-bezier(0.25, 0.1, 0.25, 1) !important;
        box-shadow: 0 2px 8px rgba(0, 113, 227, 0.25) !important;
    }

    .stButton > button:hover {
        background: var(--accent-blue-hover) !important;
        transform: scale(1.02) !important;
        box-shadow: 0 4px 16px rgba(0, 113, 227, 0.35) !important;
    }

    .stButton > button:active {
        transform: scale(0.98) !important;
    }

    /* === SECONDARY/DOWNLOAD BUTTONS === */
    .stDownloadButton > button {
        font-family: var(--font-text) !important;
        background: transparent !important;
        color: var(--accent-blue) !important;
        border: 1.5px solid var(--accent-blue) !important;
        border-radius: var(--radius-md) !important;
        padding: 0.625rem 1.25rem !important;
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }

    .stDownloadButton > button:hover {
        background: var(--accent-blue) !important;
        color: white !important;
    }

    /* === FILE UPLOADER === */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.6);
        backdrop-filter: blur(10px);
        border-radius: var(--radius-lg);
        border: 2px dashed var(--border-medium);
        padding: 2rem;
        transition: all 0.3s ease;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: var(--accent-blue);
        background: rgba(0, 113, 227, 0.04);
    }

    [data-testid="stFileUploader"] section {
        padding: 0 !important;
    }

    [data-testid="stFileUploader"] button {
        font-family: var(--font-text) !important;
        background: var(--accent-blue) !important;
        color: white !important;
        border: none !important;
        border-radius: var(--radius-sm) !important;
        font-weight: 500 !important;
    }

    /* === DATA TABLES === */
    .stDataFrame {
        border-radius: var(--radius-lg) !important;
        overflow: hidden !important;
        box-shadow: var(--shadow-sm) !important;
        border: 1px solid var(--border-light) !important;
    }

    .stDataFrame [data-testid="stDataFrameResizable"] {
        border-radius: var(--radius-lg) !important;
    }

    /* Table header */
    .stDataFrame thead th {
        background: var(--bg-secondary) !important;
        font-family: var(--font-text) !important;
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        color: var(--text-secondary) !important;
        padding: 1rem !important;
    }

    /* Table cells */
    .stDataFrame tbody td {
        font-family: var(--font-text) !important;
        font-size: 0.875rem !important;
        padding: 0.875rem 1rem !important;
        border-bottom: 1px solid var(--border-light) !important;
    }

    .stDataFrame tbody tr:hover {
        background: rgba(0, 113, 227, 0.04) !important;
    }

    /* === SELECT BOXES === */
    .stSelectbox > div > div {
        font-family: var(--font-text) !important;
        background: white !important;
        border-radius: var(--radius-md) !important;
        border: 1.5px solid var(--border-medium) !important;
        transition: all 0.2s ease !important;
    }

    .stSelectbox > div > div:hover {
        border-color: var(--accent-blue) !important;
    }

    .stSelectbox > div > div:focus-within {
        border-color: var(--accent-blue) !important;
        box-shadow: 0 0 0 3px rgba(0, 113, 227, 0.15) !important;
    }

    /* === TEXT INPUTS === */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {
        font-family: var(--font-text) !important;
        background: white !important;
        border-radius: var(--radius-sm) !important;
        border: 1.5px solid var(--border-medium) !important;
        padding: 0.625rem 0.875rem !important;
        font-size: 0.9375rem !important;
        transition: all 0.2s ease !important;
    }

    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus {
        border-color: var(--accent-blue) !important;
        box-shadow: 0 0 0 3px rgba(0, 113, 227, 0.15) !important;
        outline: none !important;
    }

    /* === TABS === */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0 !important;
        background: var(--bg-secondary);
        border-radius: var(--radius-md);
        padding: 4px;
    }

    .stTabs [data-baseweb="tab"] {
        font-family: var(--font-text) !important;
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        color: var(--text-secondary) !important;
        border-radius: var(--radius-sm) !important;
        padding: 0.5rem 1rem !important;
        background: transparent !important;
        border: none !important;
        transition: all 0.2s ease !important;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text-primary) !important;
        background: rgba(255, 255, 255, 0.5) !important;
    }

    .stTabs [aria-selected="true"] {
        background: white !important;
        color: var(--text-primary) !important;
        box-shadow: var(--shadow-sm) !important;
    }

    .stTabs [data-baseweb="tab-highlight"] {
        display: none !important;
    }

    .stTabs [data-baseweb="tab-border"] {
        display: none !important;
    }

    /* === EXPANDER === */
    .streamlit-expanderHeader {
        font-family: var(--font-text) !important;
        font-size: 0.9375rem !important;
        font-weight: 500 !important;
        color: var(--text-primary) !important;
        background: rgba(255, 255, 255, 0.6) !important;
        border-radius: var(--radius-md) !important;
        border: 1px solid var(--border-light) !important;
        padding: 1rem !important;
        transition: all 0.2s ease !important;
    }

    .streamlit-expanderHeader:hover {
        background: rgba(255, 255, 255, 0.9) !important;
        border-color: var(--border-medium) !important;
    }

    [data-testid="stExpander"] {
        background: transparent !important;
        border: none !important;
    }

    /* === ALERTS/MESSAGES === */
    .stSuccess, .stError, .stWarning, .stInfo {
        font-family: var(--font-text) !important;
        border-radius: var(--radius-md) !important;
        border: none !important;
        padding: 1rem 1.25rem !important;
    }

    .stSuccess {
        background: linear-gradient(135deg, rgba(52, 199, 89, 0.12) 0%, rgba(48, 209, 88, 0.08) 100%) !important;
        color: #248a3d !important;
    }

    .stError {
        background: linear-gradient(135deg, rgba(255, 59, 48, 0.12) 0%, rgba(255, 69, 58, 0.08) 100%) !important;
        color: #d70015 !important;
    }

    .stWarning {
        background: linear-gradient(135deg, rgba(255, 159, 10, 0.12) 0%, rgba(255, 179, 64, 0.08) 100%) !important;
        color: #b25000 !important;
    }

    .stInfo {
        background: linear-gradient(135deg, rgba(0, 113, 227, 0.12) 0%, rgba(10, 132, 255, 0.08) 100%) !important;
        color: #0071e3 !important;
    }

    /* === SIDEBAR === */
    [data-testid="stSidebar"] {
        background: rgba(245, 245, 247, 0.95) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border-right: 1px solid var(--border-light) !important;
    }

    [data-testid="stSidebar"] .block-container {
        padding: 2rem 1.5rem !important;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        font-size: 0.875rem !important;
        font-weight: 600 !important;
        color: var(--text-secondary) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
        margin-top: 1.5rem !important;
        margin-bottom: 0.75rem !important;
        background: none !important;
        -webkit-text-fill-color: var(--text-secondary) !important;
    }

    /* === DIVIDERS === */
    hr {
        border: none !important;
        height: 1px !important;
        background: linear-gradient(90deg, transparent, var(--border-medium), transparent) !important;
        margin: 2.5rem 0 !important;
    }

    /* === SPINNER === */
    .stSpinner > div {
        border-color: var(--accent-blue) transparent transparent transparent !important;
    }

    /* === CUSTOM HEADER COMPONENT === */
    .app-header {
        text-align: center;
        padding: 3.5rem 2rem 2.5rem 2rem;
        margin-bottom: 1.5rem;
        background: linear-gradient(180deg, rgba(255,255,255,0.9) 0%, rgba(245,245,247,0.5) 100%);
        border-radius: var(--radius-xl);
        border: 1px solid var(--border-light);
    }

    .app-header h1 {
        font-size: 3.25rem !important;
        margin-bottom: 0.75rem !important;
        background: linear-gradient(135deg, #1d1d1f 0%, #0071e3 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .app-header .subtitle {
        font-size: 1.375rem;
        color: var(--text-primary);
        font-weight: 500;
        letter-spacing: -0.02em;
        margin-bottom: 0.75rem;
    }

    .app-header .value-prop {
        font-size: 1rem;
        color: var(--text-secondary);
        font-weight: 400;
        max-width: 600px;
        margin: 0 auto 1.25rem auto;
        line-height: 1.6;
    }

    .app-header .badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        background: linear-gradient(135deg, rgba(52, 199, 89, 0.12) 0%, rgba(0, 113, 227, 0.12) 100%);
        color: #248a3d;
        font-size: 0.8125rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        border-radius: 100px;
        border: 1px solid rgba(52, 199, 89, 0.2);
    }

    /* === SECTION HEADERS === */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin: 2rem 0 1.25rem 0;
    }

    .section-header h3 {
        margin: 0 !important;
        padding: 0 !important;
    }

    .section-icon {
        width: 32px;
        height: 32px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, var(--accent-blue) 0%, #34aadc 100%);
        border-radius: 8px;
        font-size: 1rem;
    }

    /* === KEYWORD TAGS === */
    .keyword-tag {
        display: inline-flex;
        align-items: center;
        gap: 0.375rem;
        background: white;
        border: 1px solid var(--border-light);
        padding: 0.5rem 0.875rem;
        border-radius: 100px;
        margin: 0.25rem;
        font-size: 0.8125rem;
        font-weight: 500;
        color: var(--text-primary);
        box-shadow: var(--shadow-sm);
        transition: all 0.2s ease;
    }

    .keyword-tag:hover {
        transform: translateY(-1px);
        box-shadow: var(--shadow-md);
        border-color: var(--accent-blue);
    }

    .keyword-tag .count {
        color: var(--text-tertiary);
        font-weight: 400;
    }

    /* === FOOTER === */
    .app-footer {
        text-align: center;
        padding: 2rem 0;
        margin-top: 2rem;
    }

    .app-footer p {
        font-size: 0.8125rem;
        color: var(--text-tertiary);
        font-weight: 400;
    }

    .app-footer .dot {
        display: inline-block;
        width: 3px;
        height: 3px;
        background: var(--text-tertiary);
        border-radius: 50%;
        margin: 0 0.75rem;
        vertical-align: middle;
    }

    /* === ANIMATIONS === */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .animate-in {
        animation: fadeIn 0.5s ease forwards;
    }

    /* === SCROLLBAR === */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: transparent;
    }

    ::-webkit-scrollbar-thumb {
        background: var(--border-medium);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: var(--text-tertiary);
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* === AI ANALYSIS CARDS === */
    .ai-headline {
        font-size: 1.625rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 1.5rem;
        padding: 1.25rem 1.5rem;
        background: linear-gradient(135deg, rgba(0, 113, 227, 0.08) 0%, rgba(52, 199, 89, 0.08) 100%);
        border-radius: var(--radius-lg);
        border-left: 4px solid var(--accent-blue);
        line-height: 1.4;
    }

    .evidence-card {
        background: white;
        border-radius: var(--radius-md);
        padding: 1rem 1.25rem;
        margin: 0.5rem 0;
        border-left: 4px solid;
        box-shadow: var(--shadow-sm);
    }

    .evidence-card.data-based {
        border-color: #34c759;
        background: linear-gradient(135deg, rgba(52, 199, 89, 0.08) 0%, white 100%);
    }

    .evidence-card.indication {
        border-color: #ff9500;
        background: linear-gradient(135deg, rgba(255, 149, 0, 0.08) 0%, white 100%);
    }

    .evidence-card.open {
        border-color: #007aff;
        background: linear-gradient(135deg, rgba(0, 122, 255, 0.08) 0%, white 100%);
    }

    .evidence-label {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.25rem;
    }

    .evidence-card.data-based .evidence-label { color: #248a3d; }
    .evidence-card.indication .evidence-label { color: #b25000; }
    .evidence-card.open .evidence-label { color: #0071e3; }

    .evidence-text {
        font-size: 0.9375rem;
        color: var(--text-primary);
        line-height: 1.5;
    }

    .summary-bullet {
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
        padding: 0.5rem 0;
    }

    .summary-bullet .bullet-icon {
        width: 24px;
        height: 24px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.75rem;
        font-weight: 600;
        flex-shrink: 0;
    }

    .summary-bullet .bullet-icon.data { background: rgba(52, 199, 89, 0.15); color: #248a3d; }
    .summary-bullet .bullet-icon.indication { background: rgba(255, 149, 0, 0.15); color: #b25000; }
    .summary-bullet .bullet-icon.open { background: rgba(0, 122, 255, 0.15); color: #0071e3; }

    .drivers-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
        gap: 0.75rem;
        margin: 1rem 0;
    }

    .driver-card {
        background: white;
        border-radius: var(--radius-sm);
        padding: 0.875rem 1rem;
        border: 1px solid var(--border-light);
        box-shadow: var(--shadow-sm);
    }

    .driver-name {
        font-size: 0.8125rem;
        font-weight: 500;
        color: var(--text-secondary);
        margin-bottom: 0.25rem;
    }

    .driver-value {
        font-size: 1.125rem;
        font-weight: 600;
        color: var(--text-primary);
    }

    .driver-share {
        font-size: 0.75rem;
        color: var(--text-tertiary);
    }

    .questions-list {
        background: rgba(0, 122, 255, 0.04);
        border-radius: var(--radius-md);
        padding: 1rem 1.25rem;
        margin-top: 1rem;
    }

    .questions-list h4 {
        font-size: 0.875rem;
        font-weight: 600;
        color: var(--accent-blue);
        margin-bottom: 0.75rem;
    }

    .question-item {
        display: flex;
        align-items: flex-start;
        gap: 0.5rem;
        padding: 0.375rem 0;
        font-size: 0.9375rem;
        color: var(--text-primary);
    }

    .question-item::before {
        content: "?";
        font-weight: 600;
        color: var(--accent-blue);
    }

    /* Executive Summary Intro */
    .exec-intro {
        margin-bottom: 1.5rem;
    }

    .exec-intro h3 {
        font-size: 1.5rem !important;
        margin-bottom: 0.5rem !important;
    }

    .exec-description {
        font-size: 1rem;
        color: var(--text-secondary);
        line-height: 1.6;
        max-width: 700px;
    }

    /* Executive Summary Cards */
    .exec-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(250,250,252,0.95) 100%);
        border-radius: var(--radius-lg);
        padding: 1.75rem;
        margin: 1rem 0;
        box-shadow: var(--shadow-md);
        border: 1px solid var(--border-light);
        transition: all 0.3s ease;
    }

    .exec-card:hover {
        box-shadow: var(--shadow-lg);
        transform: translateY(-2px);
    }

    .exec-card h4 {
        font-size: 0.8125rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--accent-blue);
        margin-bottom: 1.25rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid var(--border-light);
    }

    .finding-item {
        padding: 0.75rem 0;
        border-bottom: 1px solid var(--border-light);
        font-size: 0.9375rem;
    }

    .finding-item:last-child {
        border-bottom: none;
    }

    .variance-highlight {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.75rem;
        background: var(--bg-secondary);
        border-radius: var(--radius-sm);
        margin: 0.5rem 0;
    }

    .variance-highlight .name {
        font-weight: 500;
    }

    .variance-highlight .delta {
        font-weight: 600;
        font-family: var(--font-display);
    }

    .variance-highlight .delta.positive { color: #34c759; }
    .variance-highlight .delta.negative { color: #ff3b30; }
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
if "demo_loaded" not in st.session_state:
    st.session_state.demo_loaded = False
if "executive_summary" not in st.session_state:
    st.session_state.executive_summary = None


def render_ai_analysis(data: dict, backend_name: str = ""):
    """Render AI analysis with beautiful cards and badges."""
    # Headline
    if data.get("headline"):
        st.markdown(f'<div class="ai-headline">{data["headline"]}</div>', unsafe_allow_html=True)

    # Summary bullets with evidence badges
    if data.get("summary"):
        for bullet in data["summary"]:
            # Determine evidence type from prefix
            if "[Datenbasiert]" in bullet:
                icon_class = "data"
                icon = "D"
                text = bullet.replace("[Datenbasiert]", "").strip()
            elif "[Indiz]" in bullet:
                icon_class = "indication"
                icon = "I"
                text = bullet.replace("[Indiz]", "").strip()
            elif "[Offen]" in bullet:
                icon_class = "open"
                icon = "?"
                text = bullet.replace("[Offen]", "").strip()
            else:
                icon_class = "data"
                icon = "•"
                text = bullet

            st.markdown(f'''
                <div class="summary-bullet">
                    <div class="bullet-icon {icon_class}">{icon}</div>
                    <div>{text}</div>
                </div>
            ''', unsafe_allow_html=True)

    # Key Drivers as cards
    if data.get("drivers"):
        st.markdown("**Treiber**")
        drivers_html = '<div class="drivers-grid">'
        for d in data["drivers"][:6]:
            name = d.get("name", "-")
            delta = d.get("delta", 0)
            share = d.get("share", 0)
            drivers_html += f'''
                <div class="driver-card">
                    <div class="driver-name">{name}</div>
                    <div class="driver-value">{delta:+,.0f} EUR</div>
                    <div class="driver-share">{share:.0%} Anteil</div>
                </div>
            '''
        drivers_html += '</div>'
        st.markdown(drivers_html, unsafe_allow_html=True)

    # Evidence cards
    if data.get("evidence"):
        st.markdown("**Evidenz**")
        for ev in data["evidence"]:
            label = ev.get("label", "Offen")
            text = ev.get("text", "")

            if label == "Datenbasiert":
                card_class = "data-based"
            elif label == "Indiz":
                card_class = "indication"
            else:
                card_class = "open"

            st.markdown(f'''
                <div class="evidence-card {card_class}">
                    <div class="evidence-label">{label}</div>
                    <div class="evidence-text">{text}</div>
                </div>
            ''', unsafe_allow_html=True)

    # Open questions
    if data.get("questions"):
        questions_html = '<div class="questions-list"><h4>Offene Fragen</h4>'
        for q in data["questions"]:
            questions_html += f'<div class="question-item">{q}</div>'
        questions_html += '</div>'
        st.markdown(questions_html, unsafe_allow_html=True)

    # Backend info
    if backend_name:
        st.caption(f"Analysiert mit {backend_name}")


def _strip_html(text: str) -> str:
    """Strip HTML tags from text to prevent injection from AI responses."""
    import re
    if not text:
        return ""
    # Remove HTML tags
    clean = re.sub(r'<[^>]+>', '', str(text))
    # Also escape any remaining < > to be safe
    clean = clean.replace('<', '&lt;').replace('>', '&gt;')
    return clean.strip()


def render_executive_summary(data: dict):
    """Render executive summary with styled cards."""
    # Headline
    if data.get("headline"):
        headline = _strip_html(data["headline"])
        st.markdown(f'<div class="ai-headline">{headline}</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Key findings
        if data.get("key_findings"):
            findings_html = '<div class="exec-card"><h4>Wichtigste Erkenntnisse</h4>'
            for finding in data["key_findings"]:
                if finding and isinstance(finding, str) and "<div" not in finding:
                    findings_html += f'<div class="finding-item">{_strip_html(finding)}</div>'
            findings_html += '</div>'
            st.markdown(findings_html, unsafe_allow_html=True)

        # Patterns
        if data.get("patterns"):
            patterns_html = '<div class="exec-card"><h4>Erkannte Muster</h4>'
            for pattern in data["patterns"]:
                if pattern and isinstance(pattern, str) and "<div" not in pattern:
                    patterns_html += f'<div class="finding-item">{_strip_html(pattern)}</div>'
            patterns_html += '</div>'
            st.markdown(patterns_html, unsafe_allow_html=True)

    with col2:
        # Top variances
        if data.get("top_variances"):
            variances_html = '<div class="exec-card"><h4>Top Abweichungen</h4>'
            for v in data["top_variances"]:
                # Skip if not a dict (AI sometimes returns HTML strings)
                if not isinstance(v, dict):
                    continue
                name = _strip_html(v.get("name", ""))
                if not name or "<" in name:  # Skip if still contains HTML
                    continue
                delta = v.get("delta", 0)
                if not isinstance(delta, (int, float)):
                    try:
                        delta = float(str(delta).replace(",", "").replace("+", ""))
                    except (ValueError, TypeError):
                        delta = 0
                reason = _strip_html(v.get("reason", ""))
                delta_class = "positive" if delta > 0 else "negative"
                variances_html += f'''
                    <div class="variance-highlight">
                        <div>
                            <div class="name">{name}</div>
                            <div style="font-size: 0.8125rem; color: #86868b;">{reason}</div>
                        </div>
                        <div class="delta {delta_class}">{delta:+,.0f}</div>
                    </div>
                '''
            variances_html += '</div>'
            st.markdown(variances_html, unsafe_allow_html=True)

        # Recommendations
        if data.get("recommendations"):
            recs_html = '<div class="exec-card"><h4>Empfehlungen</h4>'
            for rec in data["recommendations"]:
                if rec and isinstance(rec, str) and "<div" not in rec:
                    recs_html += f'<div class="finding-item">{_strip_html(rec)}</div>'
            recs_html += '</div>'
            st.markdown(recs_html, unsafe_allow_html=True)

    # Open items
    if data.get("open_items"):
        items_html = '<div class="questions-list"><h4>Klärungsbedarf</h4>'
        for item in data["open_items"]:
            if item and isinstance(item, str) and "<div" not in item:
                items_html += f'<div class="question-item">{_strip_html(item)}</div>'
        items_html += '</div>'
        st.markdown(items_html, unsafe_allow_html=True)

# --- Sample Data Path ---
SAMPLE_DIR = Path(__file__).parent / "sample_data"
SAMPLE_PRIOR = SAMPLE_DIR / "buchungen_Q2_2024_fiktiv.csv"
SAMPLE_CURR = SAMPLE_DIR / "buchungen_Q2_2025_fiktiv.csv"

# --- Sidebar ---
with st.sidebar:
    st.markdown("### Schnellstart")
    if SAMPLE_PRIOR.exists() and SAMPLE_CURR.exists():
        if st.button("Demo laden", type="primary", use_container_width=True):
            raw_prior = load_csv(SAMPLE_PRIOR)
            raw_curr = load_csv(SAMPLE_CURR)

            mapping = ColumnMapping(
                posting_date="posting_date",
                amount="amount",
                account="account",
                account_name="account_name",
                cost_center="cost_center",
                vendor="vendor",
                text="text",
            )

            st.session_state.prior_df = normalize(raw_prior, mapping, SignMode.AS_IS)
            st.session_state.curr_df = normalize(raw_curr, mapping, SignMode.AS_IS)
            st.session_state.variance_df = variance_by_account(
                st.session_state.prior_df, st.session_state.curr_df
            )
            st.session_state.demo_loaded = True
            st.rerun()

        st.caption("Q2 2024 vs Q2 2025 (synthetisch)")
    else:
        st.warning("Demodaten nicht gefunden")

    st.markdown("---")
    st.markdown("### Materialität")
    st.caption("Nur wesentliche Abweichungen anzeigen")
    min_abs_delta = st.number_input("Min. |Abweichung|", 0, 1000000, 10000, 1000, help="Mindestbetrag der absoluten Abweichung")
    min_pct_delta = st.number_input("Min. Abweichung %", 0, 100, 10, 5, help="Mindest-Prozentuale Änderung") / 100
    min_base = st.number_input("Min. Basiswert", 0, 1000000, 5000, 1000, help="Mindestbetrag im Vorjahr oder aktuell")
    min_share_total = st.number_input("Min. Anteil %", 0, 100, 3, 1, help="Mindestanteil an der Gesamtabweichung") / 100

    st.markdown("### Anzeige")
    sign_mode = st.selectbox(
        "Vorzeichenmodus",
        options=[SignMode.AS_IS, SignMode.INVERT, SignMode.ABS],
        format_func=lambda x: {"as_is": "Wie exportiert", "invert": "Vorzeichen umkehren", "abs": "Absolutwerte"}[x.value],
    )

    dimension = st.selectbox(
        "Treiber-Dimension",
        ["cost_center", "vendor"],
        format_func=lambda x: "Kostenstelle" if x == "cost_center" else "Lieferant",
        help="Nach welcher Dimension sollen die Treiber analysiert werden?"
    )

    st.markdown("### KI-Einstellungen")

    # Detect available backend
    ollama_config = get_ollama_config()
    ollama_url = st.text_input("Ollama URL", ollama_config["base_url"])

    backend_status = detect_backend(ollama_url=ollama_url)

    # Show backend status
    if backend_status.available:
        if backend_status.backend == Backend.OLLAMA:
            st.success(f"{backend_status.icon} {backend_status.display_name}")
            st.caption(f"Model: {ollama_config['model']}")
            ollama_model = st.text_input("Ollama Model", ollama_config["model"])
            openai_model = None
        else:  # OpenAI
            st.success(f"{backend_status.icon} {backend_status.display_name}")
            st.caption(f"Model: {backend_status.model}")
            openai_model = st.text_input("OpenAI Model", backend_status.model)
            ollama_model = None
    else:
        st.error("No AI Backend")
        st.caption("Run `ollama serve` or set OPENAI_API_KEY")
        ollama_model = None
        openai_model = None

    prompt_mode = st.selectbox(
        "Prompt-Modus",
        ["strict", "normal"],
        index=0,
        format_func=lambda x: "Strikt (empfohlen)" if x == "strict" else "Normal",
        help="Strikt = nur faktenbasierte Aussagen, Normal = mehr Interpretationen"
    )

# --- Header ---
st.markdown("""
<div class="app-header">
    <h1>Clarity</h1>
    <p class="subtitle">Quartalsabweichungen verstehen. In Sekunden statt Stunden.</p>
    <p class="value-prop">Laden Sie Ihre Buchungsdaten hoch und erhalten Sie sofort eine KI-gestützte Analyse aller wesentlichen Abweichungen - komplett lokal auf Ihrem Rechner, ohne Cloud.</p>
    <span class="badge">100% Lokal &amp; Privat</span>
</div>
""", unsafe_allow_html=True)

# --- Upload Section ---
if st.session_state.demo_loaded and st.session_state.variance_df is not None:
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        st.success(f"Demodaten geladen: {len(st.session_state.prior_df):,} + {len(st.session_state.curr_df):,} Buchungen")
    with col3:
        if st.button("Zurücksetzen", type="secondary"):
            st.session_state.prior_df = None
            st.session_state.curr_df = None
            st.session_state.variance_df = None
            st.session_state.demo_loaded = False
            st.session_state.comment_result = None
            st.session_state.executive_summary = None
            st.rerun()
else:
    st.markdown("""
    <div class="exec-intro">
        <h3>Daten importieren</h3>
        <p class="exec-description">
            Laden Sie Ihre Buchungsdaten als CSV hoch. Sie benotigen zwei Dateien:
            eine für den Vorjahreszeitraum und eine für den aktuellen Zeitraum.
        </p>
    </div>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("**Vorjahr**")
        prior_file = st.file_uploader("CSV hochladen", type=["csv"], key="prior", label_visibility="collapsed")

    with col2:
        st.markdown("**Aktueller Zeitraum**")
        curr_file = st.file_uploader("CSV hochladen", type=["csv"], key="curr", label_visibility="collapsed")

if not st.session_state.demo_loaded and 'prior_file' in dir() and prior_file and curr_file:
    raw_prior = load_csv(BytesIO(prior_file.read()))
    raw_curr = load_csv(BytesIO(curr_file.read()))

    st.success(f"Loaded: {len(raw_prior):,} + {len(raw_curr):,} rows")

    st.markdown("---")
    st.markdown("""
    <div class="exec-intro">
        <h3>Spalten-Zuordnung</h3>
        <p class="exec-description">
            Ihre CSV-Dateien haben eigene Spaltennamen. Ordnen Sie hier zu, welche Spalte
            welche Bedeutung hat - z.B. welche Spalte den Betrag enthält, welche das Konto, etc.
        </p>
    </div>
    """, unsafe_allow_html=True)

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
    if st.button("Analyze", type="primary"):
        if not map_date or not map_amount or not map_account:
            st.error("Required fields: Date, Amount, Account")
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
    st.markdown("""
    <div class="exec-intro">
        <h3>Übersicht</h3>
        <p class="exec-description">
            Ihre Daten auf einen Blick. Die Materialitätsfilter in der Seitenleiste
            helfen Ihnen, sich auf die wesentlichen Abweichungen zu konzentrieren.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Summary metrics in a clean row
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        st.metric("Vorjahr", f"{st.session_state.prior_df['amount'].sum():,.0f}")
    with c2:
        st.metric("Aktuell", f"{st.session_state.curr_df['amount'].sum():,.0f}")
    with c3:
        total_delta = st.session_state.curr_df['amount'].sum() - st.session_state.prior_df['amount'].sum()
        st.metric("Gesamtabweichung", f"{total_delta:+,.0f}")
    with c4:
        st.metric("Konten", f"{len(st.session_state.variance_df)}")

    st.markdown("")

    filtered = materiality_filter(
        st.session_state.variance_df,
        min_abs_delta=min_abs_delta if min_abs_delta > 0 else None,
        min_pct_delta=min_pct_delta if min_pct_delta > 0 else None,
        min_base=min_base if min_base > 0 else None,
        min_share_total=min_share_total if min_share_total > 0 else None,
    )

    st.caption(f"{len(filtered)} von {len(st.session_state.variance_df)} Konten angezeigt (Materialitätsfilter aktiv)")

    display = filtered.copy()
    display["prior"] = display["prior"].apply(lambda x: f"{x:,.0f}")
    display["current"] = display["current"].apply(lambda x: f"{x:,.0f}")
    display["delta"] = display["delta"].apply(lambda x: f"{x:+,.0f}")
    display["delta_pct"] = display["delta_pct"].apply(lambda x: f"{x:+.1%}" if pd.notna(x) else "—")
    display["abs_delta"] = display["abs_delta"].apply(lambda x: f"{x:,.0f}")
    display["share_of_total_abs_delta"] = display["share_of_total_abs_delta"].apply(lambda x: f"{x:.1%}")

    st.dataframe(
        display[["account", "account_name", "prior", "current", "delta", "delta_pct", "abs_delta", "share_of_total_abs_delta"]].rename(
            columns={
                "account": "Konto",
                "account_name": "Bezeichnung",
                "prior": "Vorjahr",
                "current": "Aktuell",
                "delta": "Abweichung",
                "delta_pct": "Abw. %",
                "abs_delta": "|Abweichung|",
                "share_of_total_abs_delta": "Anteil"
            }
        ),
        use_container_width=True,
        hide_index=True,
        height=320,
    )

    # --- Executive Summary Section ---
    st.markdown("---")
    st.markdown("""
    <div class="exec-intro">
        <h3>Executive Summary</h3>
        <p class="exec-description">
            Mit einem Klick analysiert die KI alle wesentlichen Abweichungen und erstellt
            eine Management-Zusammenfassung: Die wichtigsten Erkenntnisse, Muster zwischen
            Konten und konkrete Handlungsempfehlungen - fertig für Ihr nächstes Meeting.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_exec1, col_exec2 = st.columns([2, 1])
    with col_exec1:
        exec_btn = st.button("Gesamtanalyse erstellen", type="primary", key="exec_summary", help="Analysiert alle materiellen Abweichungen und erstellt eine Executive Summary")
    with col_exec2:
        if not backend_status.available:
            st.caption("KI-Backend nicht verfügbar")

    if exec_btn:
        if not backend_status.available:
            st.error("No AI backend available. Run `ollama serve` or set OPENAI_API_KEY.")
        else:
            with st.spinner(f"Analysiere {len(filtered)} Konten mit {backend_status.display_name}..."):
                # Prepare variance summary for executive analysis
                variance_summary = filtered.to_dict("records")

                # Gather additional details for top accounts (drivers, keywords)
                account_details = []
                for _, row in filtered.head(10).iterrows():
                    acc = row["account"]
                    acc_drivers = drivers_for_account(
                        st.session_state.prior_df, st.session_state.curr_df, acc, dimension
                    )
                    acc_kw = keywords_for_account(st.session_state.curr_df, acc)

                    detail = {
                        "account": acc,
                        "account_name": row.get("account_name", ""),
                        "top_drivers": [
                            {"name": d.get("cost_center") or d.get("vendor", ""), "delta": d.get("delta", 0)}
                            for d in acc_drivers.to_dict("records")[:3]
                        ] if not acc_drivers.empty else [],
                        "keywords": [k for k, _ in acc_kw[:5]] if acc_kw else [],
                    }
                    account_details.append(detail)

                # Build executive summary prompt
                total_prior = st.session_state.prior_df['amount'].sum()
                total_current = st.session_state.curr_df['amount'].sum()

                exec_prompt = format_executive_context(
                    period_info="Quartalsvergleich YoY",
                    total_prior=total_prior,
                    total_current=total_current,
                    total_delta=total_delta,
                    variance_summary=variance_summary,
                    account_details=account_details,
                )

                try:
                    response = llm_generate(
                        user_prompt=exec_prompt,
                        system_prompt=SYSTEM_PROMPT_EXECUTIVE,
                        backend=backend_status.backend,
                        ollama_url=ollama_url,
                        ollama_model=ollama_model,
                        openai_model=openai_model,
                    )
                    parsed = extract_json(response)
                    st.session_state.executive_summary = {
                        "raw": response,
                        "parsed": parsed,
                        "backend": backend_status.display_name,
                    }
                    # Debug: show parse status
                    if not parsed:
                        st.warning(f"JSON parsing failed. Response length: {len(response)} chars")
                        with st.expander("Show raw AI response"):
                            st.code(response, language="text")
                except (ConnectionError, RuntimeError) as e:
                    st.error(str(e))
                    st.session_state.executive_summary = None
                except Exception as e:
                    st.error(f"Unexpected error: {type(e).__name__}: {e}")
                    st.session_state.executive_summary = None

    # Display Executive Summary if available
    if st.session_state.executive_summary:
        exec_result = st.session_state.executive_summary

        if exec_result["parsed"]:
            exec_data = exec_result["parsed"]

            # Render executive summary
            render_executive_summary(exec_data)

            st.markdown("")

            # Download buttons for executive summary
            col_ex1, col_ex2, col_ex3 = st.columns([1, 1, 2])

            with col_ex1:
                # PDF Download
                variance_for_pdf = [
                    {
                        "account": r.get("account", ""),
                        "account_name": r.get("account_name", ""),
                        "prior": f"{r.get('prior', 0):,.0f}",
                        "current": f"{r.get('current', 0):,.0f}",
                        "delta": f"{r.get('delta', 0):+,.0f}",
                        "delta_pct": f"{r.get('delta_pct', 0):+.1%}" if r.get("delta_pct") else "-",
                        "share": f"{r.get('share_of_total_abs_delta', 0):.1%}",
                    }
                    for r in filtered.to_dict("records")[:20]
                ]

                exec_pdf = generate_executive_summary_pdf(
                    title="Varianz Executive Summary",
                    period_info="Quartalsvergleich YoY",
                    total_prior=st.session_state.prior_df['amount'].sum(),
                    total_current=st.session_state.curr_df['amount'].sum(),
                    total_delta=total_delta,
                    variance_data=variance_for_pdf,
                    executive_summary=exec_data,
                )
                st.download_button(
                    "Download PDF",
                    exec_pdf,
                    "executive_summary.pdf",
                    "application/pdf",
                    key="exec_pdf",
                )

            with col_ex2:
                # Markdown download
                exec_md = f"# Executive Summary\n\n"
                exec_md += f"## {exec_data.get('headline', '')}\n\n"
                if exec_data.get("key_findings"):
                    exec_md += "### Wichtigste Erkenntnisse\n"
                    exec_md += "\n".join(f"- {f}" for f in exec_data["key_findings"]) + "\n\n"
                if exec_data.get("top_variances"):
                    exec_md += "### Top Abweichungen\n"
                    for v in exec_data["top_variances"]:
                        exec_md += f"- **{v.get('name')}**: {v.get('delta', 0):+,.0f} EUR - {v.get('reason', '')}\n"
                    exec_md += "\n"
                if exec_data.get("recommendations"):
                    exec_md += "### Empfehlungen\n"
                    exec_md += "\n".join(f"- {r}" for r in exec_data["recommendations"]) + "\n"

                st.download_button(
                    "Download MD",
                    exec_md,
                    "executive_summary.md",
                    "text/markdown",
                    key="exec_md",
                )

            st.caption(f"Analysiert mit {exec_result.get('backend', '')}")

            with st.expander("Raw JSON"):
                st.json(exec_data)
        else:
            st.warning("Could not parse AI response")
            with st.expander("Raw response", expanded=True):
                st.code(exec_result["raw"], language=None)

    # --- Drill-Down ---
    st.markdown("---")
    st.markdown("""
    <div class="exec-intro">
        <h3>Einzelkonto-Analyse</h3>
        <p class="exec-description">
            Wählen Sie ein Konto aus, um die Treiber der Abweichung zu verstehen.
            Die KI analysiert Kostenstellen, Lieferanten und Buchungstexte, um
            mögliche Ursachen zu identifizieren.
        </p>
    </div>
    """, unsafe_allow_html=True)

    accounts = filtered["account"].tolist()
    if accounts:
        selected = st.selectbox(
            "Konto auswählen",
            accounts,
            format_func=lambda x: f"{x} - {filtered[filtered['account'] == x]['account_name'].iloc[0]}",
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
            st.metric("Prior", f"{prior_val:,.0f}")
        with c2:
            st.metric("Current", f"{curr_val:,.0f}")
        with c3:
            pct_str = f"{pct_val:+.1%}" if pd.notna(pct_val) else "—"
            st.metric("Variance", f"{delta_val:+,.0f}", delta=pct_str)

        st.markdown("")

        tab1, tab2, tab3 = st.tabs(["Treiber", "Buchungen", "Schlagwörter"])

        with tab1:
            st.caption("Welche Kostenstellen oder Lieferanten treiben die Abweichung?")
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
                st.info("Keine Treiberdaten für dieses Konto verfügbar")

        with tab2:
            st.caption("Die größten Einzelbuchungen im aktuellen Zeitraum")
            samples = samples_for_account(st.session_state.curr_df, selected)
            if not samples.empty:
                st.dataframe(samples, use_container_width=True, hide_index=True)
            else:
                st.info("Keine Buchungen für dieses Konto gefunden")

        with tab3:
            st.caption("Häufige Begriffe in den Buchungstexten")
            kw = keywords_for_account(st.session_state.curr_df, selected)
            if kw:
                kw_html = " ".join(
                    f'<span class="keyword-tag">{k} <span class="count">({c})</span></span>'
                    for k, c in kw[:15]
                )
                st.markdown(kw_html, unsafe_allow_html=True)
            else:
                st.info("Keine Schlagwörter aus Buchungstexten extrahiert")

        # --- AI Comment ---
        st.markdown("---")
        st.markdown("""
        <div class="exec-intro">
            <h3>KI-Kommentar</h3>
            <p class="exec-description">
                Lassen Sie die KI einen Kommentar für dieses Konto erstellen.
                Basierend auf Treibern, Buchungen und Schlagwörtern formuliert die KI
                eine Erklärung der Abweichung mit Evidenz-Bewertung.
            </p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([3, 1])
        with col1:
            generate_btn = st.button("KI-Kommentar erstellen", type="primary", help="Erstellt einen detaillierten Kommentar mit Ursachenanalyse")
        with col2:
            if not backend_status.available:
                st.caption("KI offline")

        if generate_btn:
            if not backend_status.available:
                st.error("No AI backend available. Run `ollama serve` or set OPENAI_API_KEY.")
            else:
                with st.spinner(f"Analyzing with {backend_status.display_name}..."):
                    drivers_list = drivers.to_dict("records") if not drivers.empty else []
                    samples_list = samples.to_dict("records") if not samples.empty else []

                    var_row = st.session_state.variance_df[st.session_state.variance_df["account"] == selected].iloc[0]
                    abs_delta_val = var_row.get("abs_delta")
                    share_total_val = var_row.get("share_of_total_abs_delta")

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
                        abs_delta=abs_delta_val,
                        share_of_total=share_total_val,
                    )

                    try:
                        system_prompt = get_system_prompt(prompt_mode)
                        response = llm_generate(
                            user_prompt=prompt,
                            system_prompt=system_prompt,
                            backend=backend_status.backend,
                            ollama_url=ollama_url,
                            ollama_model=ollama_model,
                            openai_model=openai_model,
                        )
                        st.session_state.comment_result = {
                            "raw": response,
                            "parsed": extract_json(response),
                            "prompt_mode": prompt_mode,
                            "backend": backend_status.display_name,
                        }
                    except (ConnectionError, RuntimeError) as e:
                        st.error(str(e))
                        st.session_state.comment_result = None

        if st.session_state.comment_result:
            result = st.session_state.comment_result

            if result["parsed"]:
                data = result["parsed"]

                # Use new beautiful renderer
                render_ai_analysis(data, result.get("backend", ""))

                st.markdown("")

                # Download buttons
                col_dl1, col_dl2, col_dl3 = st.columns([1, 1, 2])

                with col_dl1:
                    # PDF Download
                    pdf_bytes = generate_single_account_pdf(
                        account=selected,
                        account_name=acc_row["account_name"] or "",
                        prior=prior_val,
                        current=curr_val,
                        delta=delta_val,
                        delta_pct=pct_val,
                        analysis=data,
                        drivers=drivers.to_dict("records") if not drivers.empty else None,
                    )
                    st.download_button(
                        "Download PDF",
                        pdf_bytes,
                        f"analyse_{selected}.pdf",
                        "application/pdf",
                    )

                with col_dl2:
                    # Markdown Download
                    md = f"# {data.get('headline', 'Analysis')}\n\n"
                    md += "## Summary\n" + "\n".join(f"- {b}" for b in data.get("summary", [])) + "\n\n"
                    md += "## Drivers\n" + "\n".join(
                        f"- {d.get('name')}: {d.get('delta', 0):+,.0f}" for d in data.get("drivers", [])
                    ) + "\n"
                    st.download_button(
                        "Download MD",
                        md,
                        f"analyse_{selected}.md",
                        "text/markdown",
                    )

                with st.expander("Raw JSON"):
                    st.json(data)
            else:
                st.warning("Could not parse AI response as JSON")
                st.caption("Try using a different model or the 'Strict' prompt mode")
                with st.expander("View raw response", expanded=True):
                    st.code(result["raw"], language=None)

# --- Footer ---
st.markdown("""
<div class="app-footer">
    <p>Clarity <span class="dot"></span> Läuft 100% lokal <span class="dot"></span> Ihre Daten verlassen nie Ihren Rechner</p>
</div>
""", unsafe_allow_html=True)
