"""Naive baselines and a simple moving-average model."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .data_loader import assert_no_future, month_ordinal, shift_month


def recursive_naive_forecast(
    history_wide: pd.DataFrame,
    cutoff: str,
    forecast_months: tuple[str, ...],
    lag: int,
    destinations: tuple[str, ...],
) -> pd.DataFrame:
    """Faithful copy of the starter notebook's recursive naive forecast."""
    if lag <= 0:
        raise ValueError("lag must be a positive integer")
    assert_no_future(history_wide, cutoff, "naive history")
    available = history_wide.loc[
        history_wide["Date"] <= cutoff, ["Date", *destinations]
    ]
    values = {
        row["Date"]: pd.to_numeric(row[list(destinations)], errors="coerce").to_numpy(float)
        for _, row in available.iterrows()
    }
    output_rows = []
    for month in forecast_months:
        source_month = shift_month(month, -lag)
        if source_month not in values:
            raise ValueError(f"No lag-{lag} source for {month}: {source_month}")
        values[month] = values[source_month].copy()
        output_rows.append(
            {"Date": month, **dict(zip(destinations, values[month], strict=True))}
        )
    return pd.DataFrame(output_rows, columns=["Date", *destinations])


def fill_missing_from_history(
    forecast_wide: pd.DataFrame,
    history_wide: pd.DataFrame,
    cutoff: str,
    destinations: tuple[str, ...],
    max_years_back: int = 3,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Replace non-finite cells with the latest available same-calendar-month value.

    Chile is missing 2021M08 and 2021M09, so raw lag-12 has two undefined cells.
    The fallback looks back at the same month in earlier years, then at the last
    observed value.  Every fallback is logged.
    """
    history = history_wide.loc[history_wide["Date"] <= cutoff].copy()
    history = history.set_index("Date")
    output = forecast_wide.copy()
    fallbacks: list[dict[str, Any]] = []
    for row_index, month in enumerate(output["Date"]):
        for destination in destinations:
            value = output.at[row_index, destination]
            if np.isfinite(value):
                continue
            replacement = np.nan
            source_month = ""
            for years in range(1, max_years_back + 1):
                candidate = shift_month(month, -12 * years)
                if candidate > cutoff or candidate not in history.index:
                    continue
                candidate_value = pd.to_numeric(
                    pd.Series([history.at[candidate, destination]]), errors="coerce"
                ).iloc[0]
                if pd.notna(candidate_value) and np.isfinite(candidate_value):
                    replacement = float(candidate_value)
                    source_month = candidate
                    break
            if not np.isfinite(replacement):
                observed = pd.to_numeric(history[destination], errors="coerce").dropna()
                if observed.empty:
                    raise ValueError(
                        f"no available history to fill {month}/{destination}"
                    )
                replacement = float(observed.iloc[-1])
                source_month = "last_observed"
            output.at[row_index, destination] = replacement
            fallbacks.append(
                {
                    "model": "naive_lag12",
                    "Date": month,
                    "destination": destination,
                    "fallback_source": source_month,
                    "fallback_value": replacement,
                }
            )
    return output, fallbacks


def naive_lag1(
    history_wide: pd.DataFrame,
    cutoff: str,
    forecast_months: tuple[str, ...],
    destinations: tuple[str, ...],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    forecast = recursive_naive_forecast(
        history_wide, cutoff, forecast_months, lag=1, destinations=destinations
    )
    return forecast, {"lag": 1, "fallback_count": 0}


def naive_lag12_raw(
    history_wide: pd.DataFrame,
    cutoff: str,
    forecast_months: tuple[str, ...],
    destinations: tuple[str, ...],
) -> pd.DataFrame:
    """Notebook-exact lag-12; may contain undefined cells when history is missing."""
    return recursive_naive_forecast(
        history_wide, cutoff, forecast_months, lag=12, destinations=destinations
    )


def naive_lag12(
    history_wide: pd.DataFrame,
    cutoff: str,
    forecast_months: tuple[str, ...],
    destinations: tuple[str, ...],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = naive_lag12_raw(history_wide, cutoff, forecast_months, destinations)
    forecast, fallbacks = fill_missing_from_history(
        raw, history_wide, cutoff, destinations
    )
    return forecast, {
        "lag": 12,
        "fallback_count": len(fallbacks),
        "fallbacks": fallbacks,
    }


def moving_average_forecast(
    history_wide: pd.DataFrame,
    cutoff: str,
    forecast_months: tuple[str, ...],
    window: int,
    destinations: tuple[str, ...],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Constant forecast equal to the mean of the last `window` observations."""
    assert_no_future(history_wide, cutoff, "moving-average history")
    history = history_wide.loc[history_wide["Date"] <= cutoff].copy()
    if window <= 0:
        raise ValueError("window must be positive")
    row: dict[str, Any] = {"Date": None}
    fallbacks = 0
    for destination in destinations:
        observed = pd.to_numeric(history[destination], errors="coerce").dropna()
        if observed.empty:
            raise ValueError(f"{destination}: no observations before {cutoff}")
        recent = observed.tail(window)
        if len(recent) < window:
            fallbacks += 1
        row[destination] = float(recent.mean())
    rows = []
    for month in forecast_months:
        rows.append({"Date": month, **{d: row[d] for d in destinations}})
    forecast = pd.DataFrame(rows, columns=["Date", *destinations])
    return forecast, {"window": window, "short_window_destinations": fallbacks}


def last_observed_value(
    history_wide: pd.DataFrame, cutoff: str, destinations: tuple[str, ...]
) -> pd.Series:
    history = history_wide.loc[history_wide["Date"] <= cutoff]
    values = {}
    for destination in destinations:
        observed = pd.to_numeric(history[destination], errors="coerce").dropna()
        if observed.empty:
            raise ValueError(f"{destination}: no observations before {cutoff}")
        values[destination] = float(observed.iloc[-1])
    return pd.Series(values, dtype=float)
