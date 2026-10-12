"""Deterministic sauna analytics (filename retained by request)."""

from __future__ import annotations

import pandas as pd

from utils.data_processing import display_temperature

METRICS = {
    "Session count": ("session_count", "sessions"),
    "Total sauna minutes": ("total_minutes", "min"),
    "Average session duration": ("duration", "min"),
    "Average sauna temperature": ("temperature", "temperature"),
    "Average pre-sauna heart rate": ("pre_hr", "bpm"),
    "Average post-sauna heart rate": ("post_hr", "bpm"),
    "Average paired heart-rate change": ("paired_hr_change", "bpm"),
    "Average steam rounds": ("steam_rounds", "rounds/session"),
}
MEANS = {
    "duration": "Duration (min)",
    "temperature": "Sauna Temp",
    "pre_hr": "Heart Rate Pre-Sauna",
    "post_hr": "Heart Rate Post-Sauna",
    "paired_hr_change": "Heart-rate change (bpm)",
    "steam_rounds": "Steam rounds",
}


def stats(data, unit="Celsius"):
    result = {"session_count": len(data), "total_minutes": float(data["Duration (min)"].sum())}
    for key, col in MEANS.items():
        result[key] = None if data.empty else float(data[col].mean())
    if result["temperature"] is not None:
        result["temperature"] = float(display_temperature(result["temperature"], unit))
    result["date_coverage"] = (
        None
        if data.empty
        else [data.date.min().date().isoformat(), data.date.max().date().isoformat()]
    )
    return result


def period_start(dates, grouping):
    if grouping == "Weekly":
        return dates - pd.to_timedelta(dates.dt.weekday, unit="D")
    if grouping == "Monthly":
        return dates.dt.to_period("M").dt.to_timestamp()
    raise ValueError("Grouping must be Weekly or Monthly.")


def aggregate(data, start, end, grouping="Weekly", unit="Celsius"):
    bounds = pd.Series(pd.to_datetime([start, end]))
    left, right = period_start(bounds, grouping)
    periods = pd.date_range(left, right, freq="W-MON" if grouping == "Weekly" else "MS")
    selected = data.loc[data.date.between(pd.Timestamp(start), pd.Timestamp(end))].copy()
    selected["period"] = period_start(selected.date, grouping)
    groups = {key: frame for key, frame in selected.groupby("period")}
    rows = []
    for p in periods:
        frame = groups.get(p, selected.iloc[:0])
        finish = (
            p + pd.Timedelta(days=6) if grouping == "Weekly" else p.replace(day=p.days_in_month)
        )
        rows.append(
            {
                "period": p,
                "partial": bool(p < pd.Timestamp(start) or finish > pd.Timestamp(end)),
                **stats(frame, unit),
            }
        )
    return pd.DataFrame(rows)


def maximum_week(data, start, end):
    weekly = aggregate(data, start, end)
    maximum = int(weekly.session_count.max()) if not weekly.empty else 0
    ties = weekly.loc[weekly.session_count == maximum, ["period", "partial"]]
    return maximum, ties


def consistency(data, start, end):
    weekly = aggregate(data, start, end)
    gaps = data.date.drop_duplicates().sort_values().diff().dt.days.dropna()
    return {
        "calendar_weeks": len(weekly),
        "weeks_with_sessions": int((weekly.session_count > 0).sum()),
        "zero_session_weeks": int((weekly.session_count == 0).sum()),
        "partial_weeks": int(weekly.partial.sum()),
        "longest_gap_between_session_dates_days": None if gaps.empty else int(gaps.max()),
    }


def compare_periods(data, a_start, a_end, b_start, b_end, unit="Celsius"):
    from utils.data_processing import filter_dates

    a, b = (
        stats(filter_dates(data, a_start, a_end), unit),
        stats(filter_dates(data, b_start, b_end), unit),
    )
    changes = {key: None if a[key] is None or b[key] is None else b[key] - a[key] for key in MEANS}
    return {
        "a": {"range": [a_start, a_end], **a},
        "b": {"range": [b_start, b_end], **b},
        "b_minus_a": changes,
        "interpretation": "Changes over time are separate from within-session post-minus-pre differences; measurement timing is unknown.",
    }
