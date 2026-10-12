from datetime import date
from pathlib import Path

import pytest

from utils.data_processing import HEADERS, parse_csv


@pytest.fixture
def sample():
    result = parse_csv(
        Path("data/sample_sauna_history.csv").read_bytes(), "Celsius", today=date(2026, 10, 10)
    )
    assert not result.errors
    return result.data


@pytest.fixture
def csv_bytes():
    def make(rows, headers=None):
        return (",".join(headers or HEADERS) + "\n" + "\n".join(rows) + "\n").encode()

    return make
