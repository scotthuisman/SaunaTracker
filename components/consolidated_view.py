import pandas as pd
import plotly.express as px
import streamlit as st

from utils.llm_agent import AnalystError, fingerprint, summarize
from utils.portfolio_math import METRICS, aggregate, maximum_week


def render(data, unit, start, end, host, model, reference, today, consent):
    st.subheader("Your sauna routine")
    if data is None:
        st.info("Upload a valid CSV to explore your routine.")
        return
    signature = fingerprint(data, unit, start, end, model, reference, today, host)
    if st.session_state.get("summary_signature") != signature:
        st.session_state.summary = None
        st.session_state.summary_signature = signature
    if data.empty:
        st.info("No uploaded sessions in this date range. Adjust the shared filter.")
        return
    st.caption(f"Active range: {start} to {end} · {len(data)} sessions")
    left, right = st.columns(2)
    metric = left.selectbox("Trend metric", list(METRICS))
    grouping = right.radio("Group by", ["Weekly", "Monthly"], horizontal=True)
    column, suffix = METRICS[metric]
    if suffix == "temperature":
        suffix = "°C" if unit == "Celsius" else "°F"
    grouped = aggregate(data, start, end, grouping, unit)
    figure = px.bar(
        grouped,
        x="period",
        y=column,
        labels={
            "period": "Week beginning (Monday)" if grouping == "Weekly" else "Month beginning",
            column: f"{metric} ({suffix})",
            "session_count": "Contributing sessions",
            "partial": "Partial period",
        },
        hover_data={"session_count": True, "partial": True, "period": "|%Y-%m-%d"},
        color_discrete_sequence=["#B34D22"],
        title=f"{metric} · {grouping.lower()}",
    )
    figure.update_layout(margin=dict(l=10, r=10, t=50, b=10), bargap=0.2)
    st.plotly_chart(figure, width="stretch")
    st.caption(
        "Zero-session periods appear as zero for counts and total minutes; averages remain missing. Weekly periods run Monday–Sunday; edge periods may be partial."
    )
    count, ties = maximum_week(data, start, end)
    st.metric("Maximum sessions per week", count)
    labels = [
        f"{row.period.date()} through {(row.period + pd.Timedelta(days=6)).date()}"
        + (" (partial)" if row.partial else "")
        for row in ties.itertuples()
    ]
    st.write(("Tied weeks: " if len(labels) > 1 else "Week: ") + "; ".join(labels))
    st.caption(
        "Observed maximum within the selected range, not a recommended frequency. Weeks are grouped by their actual Monday date, including across years."
    )
    st.subheader("Paired heart-rate observations")
    table = (
        data[
            [
                "date",
                "Heart Rate Pre-Sauna",
                "Heart Rate Post-Sauna",
                "Steam rounds",
                "Heart-rate change (bpm)",
            ]
        ]
        .rename(columns={"date": "Date"})
        .copy()
    )
    table["Date"] = table["Date"].dt.strftime("%Y-%m-%d")
    st.dataframe(table, hide_index=True, width="stretch")
    st.caption(
        "Heart-rate change = post minus pre (bpm). Positive means the post-session reading was higher. Measurement timing after the session is unknown and can affect comparisons; this does not measure fitness improvement."
    )
    st.subheader("AI summary")
    action = "Refresh summary" if st.session_state.get("summary") else "Generate summary"
    if st.button(action, disabled=not host or not consent):
        with st.spinner("Generating a grounded summary…"):
            try:
                st.session_state.summary = summarize(
                    data, unit, start, end, host, model, reference, today
                )
            except AnalystError as exc:
                st.error(str(exc))
    with st.container(border=True):
        if st.session_state.get("summary"):
            st.write(st.session_state.summary)
        else:
            st.caption(
                "Generate a 2–3 sentence interpretation and cautious tracking habits for the next six calendar months. Enable AI and acknowledge the privacy notice in Settings."
            )
    if st.session_state.get("summary") and reference.get("verified"):
        st.markdown(
            f"Research context: [FoundMyFitness — Sauna]({reference['source_url']}) · retrieved {reference['retrieval_date']}"
        )
