from datetime import date

import pytest

from utils.data_processing import filter_dates, parse_csv
from utils.portfolio_math import aggregate, compare_periods, maximum_week, stats


def test_sample_independently_calculated(sample):
    september = filter_dates(sample, "2026-09-01", "2026-09-30")
    result = stats(september)
    assert result["session_count"] == 6
    assert result["total_minutes"] == 93
    assert result["duration"] == 15.5
    assert result["temperature"] == 80
    assert result["pre_hr"] == 64.5
    assert result["post_hr"] == 84.5
    assert result["paired_hr_change"] == 20
    assert result["steam_rounds"] == pytest.approx(10 / 6)


def test_weeks_cross_year_and_ties_partial(sample):
    selected = filter_dates(sample, "2025-12-28", "2026-01-05")
    weekly = aggregate(selected, "2025-12-28", "2026-01-05")
    assert list(weekly.period.dt.strftime("%Y-%m-%d")) == ["2025-12-22", "2025-12-29", "2026-01-05"]
    assert list(weekly.session_count) == [1, 4, 1]
    count, weeks = maximum_week(selected, "2025-12-31", "2026-01-01")
    assert count == 2 and weeks.partial.iloc[0]
    count, ties = maximum_week(sample, "2026-08-01", "2026-08-31")
    assert count == 1 and len(ties) == 3


def test_session_weighted_not_mean_of_period_means(csv_bytes):
    data = parse_csv(
        csv_bytes(
            ["2026-01-01,10,80,60,80,0", "2026-01-02,10,80,60,80,0", "2026-02-01,40,80,70,65,3"]
        ),
        "Celsius",
        today=date(2026, 10, 10),
    ).data
    assert stats(data)["duration"] == 20  # not (10+40)/2
    assert stats(data)["paired_hr_change"] == pytest.approx(35 / 3)
    assert list(data["Heart-rate change (bpm)"]) == [20, 20, -5]
    assert stats(data, "Fahrenheit")["temperature"] == 176


def test_empty_periods_counts_zero_averages_missing(sample):
    grouped = aggregate(sample, "2026-02-01", "2026-04-30", "Monthly")
    assert list(grouped.session_count) == [0, 0, 0]
    assert list(grouped.total_minutes) == [0, 0, 0]
    assert grouped.duration.isna().all()
    assert stats(filter_dates(sample, "2026-02-01", "2026-03-01"))["duration"] is None


def test_verified_comparison(sample):
    result = compare_periods(sample, "2026-08-01", "2026-08-31", "2026-09-01", "2026-09-30")
    assert result["a"]["session_count"] == 3 and result["b"]["session_count"] == 6
    assert result["b_minus_a"]["pre_hr"] == -0.5
    assert result["b_minus_a"]["post_hr"] == pytest.approx(84.5 - 251 / 3)
