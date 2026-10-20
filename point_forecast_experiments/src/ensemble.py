"""Simple, interpretable weighted ensembles of point forecasts."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def blend_forecasts(
    forecasts: dict[str, pd.DataFrame],
    members: list[str],
    weights: list[float],
    destinations: tuple[str, ...],
    forecast_months: tuple[str, ...],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if len(members) != len(weights):
        raise ValueError("members and weights must have the same length")
    if len(members) == 0:
        raise ValueError("at least one ensemble member is required")
    if abs(sum(weights) - 1.0) > 1e-9:
        raise ValueError(f"weights must sum to 1, got {sum(weights)}")

    stacked = []
    for member in members:
        frame = forecasts[member]
        if tuple(frame["Date"]) != tuple(forecast_months):
            raise ValueError(f"{member}: forecast months do not align")
        stacked.append(frame[list(destinations)].to_numpy(float))
    values = np.tensordot(np.asarray(weights, dtype=float), np.asarray(stacked), axes=(0, 0))
    values = np.clip(values, 0.0, None)
    if not np.isfinite(values).all():
        raise ValueError("ensemble produced non-finite values")
    rows = [
        {"Date": month, **dict(zip(destinations, values[index], strict=True))}
        for index, month in enumerate(forecast_months)
    ]
    frame = pd.DataFrame(rows, columns=["Date", *destinations])
    info = {"members": members, "weights": [float(w) for w in weights]}
    return frame, info
