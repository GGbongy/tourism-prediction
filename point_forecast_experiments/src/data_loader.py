"""Data loading and temporal isolation for the point-forecasting experiments.

The loader never returns locked-audit or final-period actuals.  Only the
training history (<= 2022M07) and the seven validation months are exposed to
the modelling code.
"""

from __future__ import annotations

import hashlib
import io
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

EXPECTED_DESTINATIONS: tuple[str, ...] = (
    "Canada",
    "Chile",
    "Mexico",
    "Taiwan China",
    "Hong Kong China",
    "Japan",
    "South Korea",
    "Macao China",
    "Maldives",
    "Cambodia",
    "Indonesia",
    "Singapore",
    "New Zealand",
    "USA",
    "Thailand",
    "Turkey",
    "Australia",
    "Hawaii",
    "Austria",
    "Czech",
)


def month_ordinal(label: str) -> int:
    """Convert a YYYYM label to a zero-based month ordinal."""
    return int(label[:4]) * 12 + int(label[-2:]) - 1


def label_from_ordinal(value: int) -> str:
    year, month0 = divmod(int(value), 12)
    return f"{year}M{month0 + 1:02d}"


def shift_month(label: str, offset: int) -> str:
    return label_from_ordinal(month_ordinal(label) + int(offset))


def months_between(start: str, end: str) -> tuple[str, ...]:
    return tuple(
        label_from_ordinal(ordinal)
        for ordinal in range(month_ordinal(start), month_ordinal(end) + 1)
    )


def assert_no_future(frame: pd.DataFrame, cutoff: str, label: str) -> None:
    """Fail loudly if a frame used for fitting contains later months."""
    if "Date" not in frame.columns:
        raise ValueError(f"{label}: no Date column to check")
    cutoff_ordinal = month_ordinal(cutoff)
    later = frame["Date"].map(month_ordinal) > cutoff_ordinal
    if bool(later.any()):
        offenders = frame.loc[later, "Date"].drop_duplicates().tolist()[:5]
        raise ValueError(f"{label}: found dates after {cutoff}: {offenders}")


@dataclass
class TourismData:
    history: pd.DataFrame
    validation_actuals: pd.DataFrame
    destinations: tuple[str, ...]
    validation_months: tuple[str, ...]
    metadata: dict[str, Any]


def _read_source(config: dict[str, Any], project_root: Path) -> tuple[bytes, str]:
    env_var = config["data"].get("env_var", "TOURISM_DATA_PATH")
    configured = os.environ.get(env_var, "").strip()
    if configured:
        path = Path(configured).expanduser()
        if not path.is_absolute():
            path = (project_root / path).resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"{env_var} points to a missing file: {path}"
            )
        return path.read_bytes(), f"env:{env_var}:{path}"

    local_path = project_root / config["data"]["local_filename"]
    if local_path.is_file():
        return local_path.read_bytes(), f"local:{local_path}"

    url = config["data"]["url"]
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "tourism-point-forecast-experiments/1.0"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read(), f"url:{url}"


def _missing_report(history: pd.DataFrame, destinations: tuple[str, ...]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for destination in destinations:
        series = pd.to_numeric(history[destination], errors="coerce")
        valid_index = series.first_valid_index()
        internal_gap_dates: list[str] = []
        if valid_index is not None:
            tail = series.loc[valid_index:]
            gap_index = tail[tail.isna()].index
            internal_gap_dates = history.loc[gap_index, "Date"].astype(str).tolist()
        report[destination] = {
            "missing_before_cutoff": int(series.isna().sum()),
            "first_valid": str(history.loc[valid_index, "Date"]) if valid_index is not None else None,
            "last_valid": str(history.loc[series.last_valid_index(), "Date"])
            if series.last_valid_index() is not None
            else None,
            "internal_gap_count": len(internal_gap_dates),
            "internal_gap_dates": internal_gap_dates,
        }
    return report


def load_data(config: dict[str, Any], project_root: Path) -> TourismData:
    raw_bytes, source = _read_source(config, project_root)
    sha256 = hashlib.sha256(raw_bytes).hexdigest()

    frame = pd.read_csv(io.BytesIO(raw_bytes))
    if frame.columns[0] != "Date":
        raise ValueError(f"first column must be Date, found {frame.columns[0]!r}")

    destinations = tuple(frame.columns[1:])
    if destinations != EXPECTED_DESTINATIONS:
        raise ValueError(
            "destination columns/order do not match the published contract: "
            f"{destinations}"
        )
    if not frame["Date"].is_unique:
        raise ValueError("Date values must be unique")
    frame = frame.sort_values("Date", key=lambda s: s.map(month_ordinal)).reset_index(drop=True)

    split = config["split"]
    history = frame.loc[frame["Date"] <= split["train_end"]].copy()
    validation_months = months_between(
        split["validation_start"], split["validation_end"]
    )
    validation_actuals = frame.loc[
        frame["Date"].isin(validation_months), ["Date", *destinations]
    ].copy()

    if len(validation_actuals) != len(validation_months):
        raise ValueError("validation months are incomplete in the source data")
    if validation_actuals[list(destinations)].isna().any().any():
        raise ValueError("validation actuals contain missing values")
    if len(history) == 0:
        raise ValueError("training history is empty")

    # The returned object deliberately excludes audit/final actuals.
    if month_ordinal(history["Date"].max()) > month_ordinal(split["train_end"]):
        raise AssertionError("history contains dates after the training cutoff")

    metadata = {
        "source": source,
        "sha256": sha256,
        "raw_shape": list(frame.shape),
        "history_shape": list(history.shape),
        "validation_shape": list(validation_actuals.shape),
        "raw_date_range": [frame["Date"].iloc[0], frame["Date"].iloc[-1]],
        "history_date_range": [history["Date"].iloc[0], history["Date"].iloc[-1]],
        "validation_months": list(validation_months),
        "destinations": list(destinations),
        "missing_report": _missing_report(history, destinations),
    }
    return TourismData(
        history=history,
        validation_actuals=validation_actuals,
        destinations=destinations,
        validation_months=validation_months,
        metadata=metadata,
    )
