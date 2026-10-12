from datetime import date

import pandas as pd
import pytest

from utils.data_processing import HEADERS, display_temperature, filter_dates, parse_csv

TODAY = date(2026, 10, 10)


def test_header_bom_whitespace_extras_preview(csv_bytes):
    payload = b"\xef\xbb\xbf" + csv_bytes(
        ["2026-09-01,15,176,60,80,2,private note"], [" " + h + " " for h in HEADERS] + ["notes"]
    )
    result = parse_csv(payload, "Fahrenheit", today=TODAY)
    assert result.errors == []
    assert result.data["Sauna Temp"].iloc[0] == pytest.approx(80)
    assert "notes" in result.preview and "notes" not in result.data
    assert display_temperature(80, "Fahrenheit") == 176


@pytest.mark.parametrize(
    "row,reason",
    [
        ("2026/09/01,15,80,60,80,2", "ISO"),
        ("09-01-2026,15,80,60,80,2", "ISO"),
        ("2026-02-30,15,80,60,80,2", "ISO"),
        ("2026-10-11,15,80,60,80,2", "future"),
        ("2026-09-01,,80,60,80,2", "missing"),
        ("2026-09-01,0,80,60,80,2", "greater than zero"),
        ("2026-09-01,15,NaN,60,80,2", "finite"),
        ("2026-09-01,15,inf,60,80,2", "finite"),
        ("2026-09-01,15,80,0,80,2", "greater than zero"),
        ("2026-09-01,15,80,60,-1,2", "greater than zero"),
        ("2026-09-01,15,80,60,80,1.5", "integer"),
        ("2026-09-01,15,80,60,80,-1", "integer"),
        ("2026-09-01,abc,80,60,80,2", "numeric"),
    ],
)
def test_invalid_values_reject_entire_upload(csv_bytes, row, reason):
    result = parse_csv(csv_bytes(["2026-08-01,15,80,60,80,2", row]), "Celsius", today=TODAY)
    assert result.data is None
    assert any("CSV row 3" in e and reason in e for e in result.errors)


@pytest.mark.parametrize(
    "payload,reason",
    [
        (b"", "empty"),
        (b"date\n2026-01-01", "missing columns"),
        (b"date,date", "duplicate"),
        (b"\xff", "UTF-8"),
    ],
)
def test_schema_errors(payload, reason):
    result = parse_csv(payload, "Celsius", today=TODAY)
    assert result.data is None
    assert any(reason in e for e in result.errors)


def test_malformed_and_no_rows(csv_bytes):
    for payload in (
        csv_bytes([]),
        csv_bytes(["2026-09-01,15,80,60,80"]),
        csv_bytes(['"2026-09-01,15,80,60,80,2']),
    ):
        result = parse_csv(payload, "Celsius", today=TODAY)
        assert result.errors and result.data is None


def test_duplicates_retained_or_removed_multiple_same_day(csv_bytes):
    rows = ["2026-09-07,18,80,60,80,2", "2026-09-07,18,80,60,80,2", "2026-09-07,10,75,62,79,1"]
    kept = parse_csv(csv_bytes(rows), "Celsius", today=TODAY)
    removed = parse_csv(csv_bytes(rows), "Celsius", True, TODAY)
    assert len(kept.data) == 3 and len(removed.data) == 2
    assert "CSV rows 3" in kept.warnings[0]
    assert len(removed.preview) == 3


def test_sort_and_empty_filter(csv_bytes):
    result = parse_csv(
        csv_bytes(["2026-09-02,15,80,60,80,2", "2026-09-01,10,75,62,79,1"]), "Celsius", today=TODAY
    )
    assert result.data.date.is_monotonic_increasing
    assert filter_dates(result.data, "2026-08-01", "2026-08-31").empty
    pd.testing.assert_frame_equal(
        filter_dates(result.data, "2026-09-01", "2026-09-02"), result.data
    )


def test_integer_validation_does_not_round_fractional_rounds(csv_bytes):
    result = parse_csv(
        csv_bytes(["2026-09-01,15,80,60,80,1.0000000000000001"]), "Celsius", today=TODAY
    )
    assert result.data is None
    assert "nonnegative integer" in result.errors[0]
