"""Sauna Fitness Guru: shared settings/state and exactly four tabs."""

import os
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from components import ai_analyst, consolidated_view, data_upload, historical_performance
from utils.data_processing import filter_dates
from utils.llm_agent import DEFAULT_HOST, DEFAULT_MODEL, load_reference

load_dotenv()
st.set_page_config(page_title="Sauna Fitness Guru", page_icon="♨️", layout="wide")
st.title("♨️ Sauna Fitness Guru")
st.write("Understand your sauna routine. Track consistently. Review how you feel.")
st.caption(
    "For informational use only. Session logs do not diagnose conditions, measure cardiovascular fitness improvement, or predict health outcomes."
)
try:
    timezone = ZoneInfo(os.getenv("APP_TIMEZONE", "America/Edmonton"))
except ZoneInfoNotFoundError:
    timezone = ZoneInfo("America/Edmonton")
    st.warning("APP_TIMEZONE is invalid; using America/Edmonton.")
today = datetime.now(timezone).date()
reference = load_reference()
try:
    secret_host = st.secrets.get("OLLAMA_HOST", DEFAULT_HOST)
    secret_model = st.secrets.get("OLLAMA_MODEL", DEFAULT_MODEL)
except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
    secret_host, secret_model = DEFAULT_HOST, DEFAULT_MODEL
host = (os.getenv("OLLAMA_HOST") or secret_host or DEFAULT_HOST).strip()
with st.sidebar:
    st.header("Settings")
    unit = st.selectbox("Display temperatures in", ["Celsius", "Fahrenheit"])
    host = st.text_input("Ollama host", value=host).strip()
    model = (
        st.text_input("Ollama model", value=os.getenv("OLLAMA_MODEL") or secret_model).strip()
        or DEFAULT_MODEL
    )
    st.caption("Local Ollama · no API key required")
    st.info(
        "Before an AI request: relevant session measurements, computed metrics, and chat messages will be sent to Ollama on the machine running this app. Uploaded files stay in this session's memory; clearing data resets chat and summaries."
    )
    consent = st.checkbox("I understand and enable AI requests", disabled=not host)
    if not reference.get("verified"):
        st.warning(reference["notice"])
    else:
        st.caption(
            f"Research snapshot: {reference['retrieval_date']} · page updated {reference.get('page_update_date', 'unknown')}"
        )

tabs = st.tabs(["Data Upload", "Consolidated View", "Historical Performance", "AI Analyst (Chat)"])
with tabs[0]:
    data_upload.render(today)
data = st.session_state.get("data")
selected, start, end = None, None, None
if data is not None:
    with st.sidebar:
        st.subheader("Chart and average date range")
        date_range = st.date_input(
            "Shared filter · Tabs 2 and 3",
            value=(data.date.min().date(), data.date.max().date()),
            min_value=pd.Timestamp.min.ceil("D").date(),
            max_value=today,
            key="date_range",
        )
        if len(date_range) == 2:
            start, end = date_range
            selected = filter_dates(data, start, end)
            st.caption(f"{start} to {end} · {len(selected)} sessions")
        else:
            st.session_state.summary = None
            st.info("Select both a start and end date.")
with tabs[1]:
    if data is not None and start is None:
        st.info("Complete the shared date range to view trends.")
    else:
        consolidated_view.render(selected, unit, start, end, host, model, reference, today, consent)
with tabs[2]:
    if data is not None and start is None:
        st.info("Complete the shared date range to view averages.")
    else:
        historical_performance.render(selected, unit, start, end)
with tabs[3]:
    ai_analyst.render(data, unit, host, model, reference, today, consent)
