"""Strict CSV validation. No personal data is written to disk or cached."""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation

import numpy as np
import pandas as pd

HEADERS = [
    "date",
    "Duration (min)",
    "Sauna Temp",
    "Heart Rate Pre-Sauna",
    "Heart Rate Post-Sauna",
    "Steam rounds",
]


@dataclass
class UploadResult:
    preview: pd.DataFrame | None = None
    data: pd.DataFrame | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def to_celsius(values, unit: str):
    if unit not in ("Celsius", "Fahrenheit"):
        raise ValueError("Choose Celsius or Fahrenheit.")
    return (values - 32) / 9 * 5 if unit == "Fahrenheit" else values


def display_temperature(values, unit: str):
    if unit not in ("Celsius", "Fahrenheit"):
        raise ValueError("Choose Celsius or Fahrenheit.")
    return values * 9 / 5 + 32 if unit == "Fahrenheit" else values


def parse_csv(
    payload: bytes, unit: str, remove_duplicates: bool = False, today: date | None = None
) -> UploadResult:
    result = UploadResult()
    today = today or date.today()
    to_celsius(0, unit)
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError:
        result.errors.append("CSV row 1: file must use UTF-8 encoding.")
        return result
    if not text.strip():
        result.errors.append("CSV row 1: empty file.")
        return result
    records, row_numbers = [], []
    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    try:
        headers = [h.strip() for h in next(reader)]
        if len(set(headers)) != len(headers):
            result.errors.append("CSV row 1: duplicate column headers after trimming whitespace.")
        missing = [h for h in HEADERS if h not in headers]
        if missing:
            result.errors.append("CSV row 1: missing columns: " + ", ".join(missing))
        for row in reader:
            number = reader.line_num
            if len(row) != len(headers):
                result.errors.append(
                    f"CSV row {number}: expected {len(headers)} fields, got {len(row)}."
                )
            else:
                records.append(row)
                row_numbers.append(number)
    except (csv.Error, StopIteration) as exc:
        result.errors.append(f"CSV row {max(reader.line_num, 1)}: malformed CSV ({exc}).")
    result.preview = pd.DataFrame(records, columns=headers)
    if result.errors:
        return result
    if not records:
        result.errors.append("CSV row 2: no session rows.")
        return result
    parsed = []
    for row_num, record in zip(row_numbers, records):
        raw = dict(zip(headers, record))
        values = {}
        raw_date = raw["date"]
        try:
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw_date):
                raise ValueError
            session_date = date.fromisoformat(raw_date)
            # pandas nanosecond dates are used throughout the analytics.
            pd.Timestamp(session_date).as_unit("ns")
            if session_date > today:
                result.errors.append(f"CSV row {row_num}: date is in the future ({raw_date}).")
            values["date"] = pd.Timestamp(session_date)
        except (ValueError, OverflowError, pd.errors.OutOfBoundsDatetime):
            result.errors.append(
                f"CSV row {row_num}: date must be a valid supported ISO YYYY-MM-DD date."
            )
        for col in HEADERS[1:]:
            try:
                if not raw[col].strip():
                    raise ValueError("missing required value")
                number = float(raw[col])
                if not np.isfinite(number):
                    raise ValueError("must be finite")
                if (
                    col in ("Duration (min)", "Heart Rate Pre-Sauna", "Heart Rate Post-Sauna")
                    and number <= 0
                ):
                    raise ValueError("must be greater than zero")
                if col == "Steam rounds":
                    exact = Decimal(raw[col])
                    if exact < 0 or exact != exact.to_integral_value():
                        raise ValueError("must be a nonnegative integer")
                    values[col] = int(exact)
                else:
                    values[col] = number
            except (ValueError, InvalidOperation) as exc:
                reason = (
                    str(exc) if not str(exc).startswith("could not convert") else "must be numeric"
                )
                result.errors.append(f"CSV row {row_num}: {col} {reason}.")
        parsed.append(values)
    if result.errors:
        return result
    # Exact raw rows include extra columns; extra columns never enter analytics/AI.
    duplicates = result.preview.duplicated()
    if duplicates.any():
        duplicate_rows = [str(row_numbers[i]) for i in range(len(records)) if duplicates.iloc[i]]
        action = "removed" if remove_duplicates else "retained"
        result.warnings.append(
            f"Exact duplicates at CSV rows {', '.join(duplicate_rows)}; {action}."
        )
    data = pd.DataFrame(parsed)
    if remove_duplicates:
        data = data.loc[~duplicates.to_numpy()].copy()
    data["Sauna Temp"] = to_celsius(data["Sauna Temp"], unit)
    data["Heart-rate change (bpm)"] = data["Heart Rate Post-Sauna"] - data["Heart Rate Pre-Sauna"]
    result.data = data.sort_values("date", kind="stable").reset_index(drop=True)
    return result


def filter_dates(data: pd.DataFrame, start, end) -> pd.DataFrame:
    return data.loc[data["date"].between(pd.Timestamp(start), pd.Timestamp(end))].copy()
