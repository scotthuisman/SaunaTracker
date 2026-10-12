import streamlit as st

from utils.portfolio_math import stats


def render(data, unit, start, end):
    st.subheader("Session-weighted averages")
    if data is None:
        st.info("Upload a valid CSV in Data Upload to view historical averages.")
        return
    if data.empty:
        st.info("No uploaded sessions in this date range. Adjust the shared filter.")
        return
    values = stats(data, unit)
    symbol = "°C" if unit == "Celsius" else "°F"
    cards = [
        ("Duration", "duration", "min"),
        ("Sauna temperature", "temperature", symbol),
        ("Heart Rate Pre-Sauna", "pre_hr", "bpm"),
        ("Heart Rate Post-Sauna", "post_hr", "bpm"),
        ("Steam rounds", "steam_rounds", "rounds/session"),
    ]
    for column, (label, key, suffix) in zip(st.columns(3), cards[:3]):
        column.metric(label, f"{values[key]:.1f} {suffix}")
    for column, (label, key, suffix) in zip(st.columns(2), cards[3:]):
        column.metric(label, f"{values[key]:.1f} {suffix}")
    st.caption(
        f"{len(data)} contributing sessions · selected range {start} to {end}. Arithmetic means across individual sessions; displayed to one decimal, calculated with full precision."
    )
