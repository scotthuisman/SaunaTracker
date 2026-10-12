"""CSV-only ingestion, with transactional replacement of session state."""

from pathlib import Path

import streamlit as st

from utils.data_processing import HEADERS, parse_csv


def clear_data():
    for key in (
        "data",
        "active_name",
        "active_input_unit",
        "preview_result",
        "preview_name",
        "date_range",
        "summary",
        "summary_signature",
    ):
        st.session_state.pop(key, None)
    st.session_state.chat_history = []
    st.session_state.upload_nonce = st.session_state.get("upload_nonce", 0) + 1


def render(today):
    st.subheader("Your history, from CSV")
    st.write(
        "Upload one CSV. Each row is one session; multiple sessions on a date are welcome. Processing a valid upload replaces the active history."
    )
    with st.expander("CSV format and validation rules", expanded=True):
        st.code(",".join(HEADERS), language="text")
        st.write(
            "Dates: YYYY-MM-DD, no future dates. Duration: positive minutes. Heart rates: positive bpm. Temperature: finite numeric values in your selected unit. Steam rounds: nonnegative integers. All six values are required; extra columns are ignored for analytics and AI."
        )
        st.caption(
            "Headers are trimmed and a UTF-8 BOM is supported. Invalid values are never discarded or repaired. Exact duplicates are kept unless you choose removal."
        )
    sample = Path(__file__).resolve().parents[1] / "data/sample_sauna_history.csv"
    st.download_button(
        "Download synthetic example CSV (Celsius)",
        sample.read_bytes(),
        "sample_sauna_history.csv",
        "text/csv",
    )
    nonce = st.session_state.get("upload_nonce", 0)
    upload = st.file_uploader(
        "Session history (.csv)", type=["csv"], accept_multiple_files=False, key=f"upload_{nonce}"
    )
    unit = st.selectbox(
        "Interpret uploaded Sauna Temp as", ["Celsius", "Fahrenheit"], key=f"input_unit_{nonce}"
    )
    remove = st.checkbox("Remove exact duplicates", key=f"dedupe_{nonce}")
    st.caption(
        "Upload settings take effect when you process the file. The active history keeps its previous interpretation until a valid replacement succeeds."
    )
    if st.button("Validate and use CSV", disabled=upload is None, type="primary"):
        result = parse_csv(upload.getvalue(), unit, remove, today)
        st.session_state.preview_result = result
        st.session_state.preview_name = upload.name
        if result.data is not None:
            st.session_state.data = result.data
            st.session_state.active_name = upload.name
            st.session_state.active_input_unit = unit
            st.session_state.chat_history = []
            st.session_state.pop("date_range", None)
            st.session_state.pop("summary", None)
            st.session_state.pop("summary_signature", None)
    result = st.session_state.get("preview_result")
    if result is not None:
        st.write(f"Validation: {st.session_state.preview_name}")
        for error in result.errors:
            st.error(error)
        for warning in result.warnings:
            st.warning(warning)
        if result.data is not None:
            st.success(
                f"Validated {len(result.data)} sessions, {result.data.date.min().date()} through {result.data.date.max().date()}."
            )
        else:
            st.warning(
                "Replacement rejected. The active dataset below remains in use."
                if "data" in st.session_state
                else "No valid dataset is active."
            )
        if result.preview is not None:
            st.caption(
                "Read-only preview of the attempted upload, preserving all original values and extra columns."
            )
            st.dataframe(result.preview, hide_index=True, width="stretch")
    if "data" in st.session_state:
        data = st.session_state.data
        st.info(
            f"Active: {st.session_state.active_name} · {len(data)} sessions · {data.date.min().date()} to {data.date.max().date()} · uploaded temperatures interpreted as {st.session_state.active_input_unit}."
        )
        st.button("Clear uploaded data", on_click=clear_data)
    else:
        st.info(
            "Upload and validate your CSV to unlock the analytical views. The example is synthetic and is never loaded automatically."
        )
