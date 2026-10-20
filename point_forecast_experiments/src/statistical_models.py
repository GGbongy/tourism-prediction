"""Interpretable statistical point-forecast models."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .data_loader import assert_no_future, month_ordinal, shift_month


def _series_map(history: pd.DataFrame, destination: str) -> dict[str, float]:
    series = pd.to_numeric(history[destination], errors="coerce")
    return {
        date: float(value)
        for date, value in zip(history["Date"], series, strict=True)
        if pd.notna(value) and np.isfinite(value)
    }


def damped_trend_forecast(
    history_wide: pd.DataFrame,
    cutoff: str,
    forecast_months: tuple[str, ...],
    window: int,
    damping: float,
    trend_cap: float,
    max_total_log_growth: float | None,
    min_points: int,
    destinations: tuple[str, ...],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Damped linear trend on log1p scale.

    The level is anchored at the last observed value before the cutoff; the
    monthly log growth is estimated by OLS over the last `window` observations
    and clipped to +/- `trend_cap`.  The cumulative log change is capped by
    `max_total_log_growth`.
    """
    assert_no_future(history_wide, cutoff, "damped-trend history")
    history = history_wide.loc[history_wide["Date"] <= cutoff].copy()
    if not (0 < damping <= 1):
        raise ValueError("damping must be in (0, 1]")

    forecasts: dict[str, list[float]] = {}
    slopes: dict[str, float] = {}
    fallbacks: list[str] = []
    capped_slopes = 0
    for destination in destinations:
        observed = history[["Date", destination]].dropna().copy()
        observed[destination] = pd.to_numeric(observed[destination], errors="coerce")
        observed = observed.dropna()
        if observed.empty:
            raise ValueError(f"{destination}: no observations before {cutoff}")

        recent = observed.tail(window)
        if len(recent) < min_points:
            slope = 0.0
            fallbacks.append(destination)
        else:
            x = recent["Date"].map(month_ordinal).to_numpy(float)
            x = x - x[0]
            y = np.log1p(recent[destination].to_numpy(float))
            slope = float(np.polyfit(x, y, 1)[0])
            clipped = float(np.clip(slope, -trend_cap, trend_cap))
            if clipped != slope:
                capped_slopes += 1
            slope = clipped
        slopes[destination] = slope

        # Prefer the value exactly at the cutoff; otherwise the last observation.
        at_cutoff = observed.loc[observed["Date"] == cutoff, destination]
        base = float(at_cutoff.iloc[0]) if len(at_cutoff) else float(observed[destination].iloc[-1])
        values = []
        for horizon, _month in enumerate(forecast_months, start=1):
            damped_steps = sum(damping ** step for step in range(1, horizon + 1))
            log_delta = slope * damped_steps
            if max_total_log_growth is not None:
                log_delta = float(
                    np.clip(-max_total_log_growth, log_delta, max_total_log_growth)
                )
            value = float(np.expm1(np.log1p(base) + log_delta))
            if not np.isfinite(value):
                value = base
            values.append(max(0.0, value))
        forecasts[destination] = values

    rows = []
    for index, month in enumerate(forecast_months):
        rows.append(
            {"Date": month, **{d: forecasts[d][index] for d in destinations}}
        )
    forecast = pd.DataFrame(rows, columns=["Date", *destinations])
    info = {
        "window": window,
        "damping": damping,
        "trend_cap": trend_cap,
        "max_total_log_growth": max_total_log_growth,
        "mean_estimated_slope": float(np.mean(list(slopes.values()))),
        "median_estimated_slope": float(np.median(list(slopes.values()))),
        "capped_slope_destinations": capped_slopes,
        "fallback_destinations": fallbacks,
    }
    return forecast, info


def adaptive_seasonal_forecast(
    history_wide: pd.DataFrame,
    cutoff: str,
    forecast_months: tuple[str, ...],
    window: int,
    ratio_cap: tuple[float, float],
    min_pairs: int,
    fallback_years: int,
    destinations: tuple[str, ...],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Seasonal same-month forecast multiplied by a recent level adjustment.

    level_adjustment = mean(recent `window` months) /
                       mean(same months one year earlier)

    The ratio is clipped to `ratio_cap`.  When the same-month value 12 months
    earlier is missing, earlier years of the same month are used; otherwise the
    recent level is used.  All fallbacks are counted.
    """
    assert_no_future(history_wide, cutoff, "adaptive-seasonal history")
    history = history_wide.loc[history_wide["Date"] <= cutoff].copy()
    low_cap, high_cap = ratio_cap
    if not (0 < low_cap <= high_cap):
        raise ValueError("invalid ratio_cap")

    forecasts: dict[str, list[float]] = {}
    ratios: dict[str, float] = {}
    base_fallbacks: list[dict[str, str]] = []
    ratio_fallbacks: list[str] = []

    recent_offsets = range(0, window)
    for destination in destinations:
        values = _series_map(history, destination)
        if not values:
            raise ValueError(f"{destination}: no observations before {cutoff}")

        recent_months = [shift_month(cutoff, -offset) for offset in recent_offsets]
        previous_months = [shift_month(month, -12) for month in recent_months]
        pairs = [
            (values[recent], values[previous])
            for recent, previous in zip(recent_months, previous_months, strict=True)
            if recent in values and previous in values
        ]
        if len(pairs) >= min_pairs:
            recent_mean = float(np.mean([pair[0] for pair in pairs]))
            previous_mean = float(np.mean([pair[1] for pair in pairs]))
            raw_ratio = (
                recent_mean / previous_mean
                if previous_mean > 0 and np.isfinite(previous_mean)
                else 1.0
            )
        else:
            raw_ratio = 1.0
            ratio_fallbacks.append(destination)
        ratio = float(np.clip(raw_ratio, low_cap, high_cap))
        ratios[destination] = ratio

        recent_values = [values[m] for m in recent_months if m in values]
        recent_level = float(np.mean(recent_values)) if recent_values else float(
            max(values.items(), key=lambda item: month_ordinal(item[0]))[1]
        )

        destination_forecasts = []
        for month in forecast_months:
            base = None
            source = None
            for years in range(1, fallback_years + 1):
                candidate = shift_month(month, -12 * years)
                if candidate <= cutoff and candidate in values:
                    base = values[candidate]
                    source = candidate
                    break
            if base is None:
                base = recent_level
                source = "recent_level"
                base_fallbacks.append(
                    {"destination": destination, "target_month": month}
                )
            value = float(base * ratio)
            if not np.isfinite(value):
                value = recent_level
            destination_forecasts.append(max(0.0, value))
        forecasts[destination] = destination_forecasts

    rows = []
    for index, month in enumerate(forecast_months):
        rows.append(
            {"Date": month, **{d: forecasts[d][index] for d in destinations}}
        )
    forecast = pd.DataFrame(rows, columns=["Date", *destinations])
    info = {
        "window": window,
        "ratio_cap": list(ratio_cap),
        "min_pairs": min_pairs,
        "fallback_years": fallback_years,
        "mean_ratio": float(np.mean(list(ratios.values()))),
        "median_ratio": float(np.median(list(ratios.values()))),
        "ratio_fallback_destinations": ratio_fallbacks,
        "base_fallback_count": len(base_fallbacks),
        "base_fallbacks": base_fallbacks,
    }
    return forecast, info
