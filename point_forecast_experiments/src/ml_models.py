"""Global machine-learning models built from cutoff-safe supervised samples.

Every training sample is (destination, forecast_origin, horizon).  Features use
only observations at or before the forecast origin, and the training target
month is never later than the training cutoff.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data_loader import assert_no_future, month_ordinal, shift_month

FEATURE_COLUMNS: list[str] = [
    "horizon",
    "month_sin",
    "month_cos",
    "last_log",
    "lag3_log",
    "lag6_log",
    "lag12_log",
    "roll3_log",
    "roll6_log",
    "roll12_log",
    "recent_change_3",
    "recent_change_6",
    "growth3",
    "growth6",
    "seasonal_rel_12",
    "missing_lag3",
    "missing_lag6",
    "missing_lag12",
]


def _series_map(history: pd.DataFrame, destination: str) -> dict[str, float]:
    series = pd.to_numeric(history[destination], errors="coerce")
    return {
        date: float(value)
        for date, value in zip(history["Date"], series, strict=True)
        if pd.notna(value) and np.isfinite(value)
    }


def _rolling_mean(values: dict[str, float], origin: str, window: int) -> float:
    months = [shift_month(origin, -offset) for offset in range(0, window)]
    available = [values[m] for m in months if m in values]
    return float(np.mean(available)) if available else np.nan


def _slope(values: dict[str, float], origin: str, window: int) -> float:
    months = [shift_month(origin, -offset) for offset in range(0, window)]
    points = [
        (month_ordinal(month), values[month])
        for month in months
        if month in values
    ]
    if len(points) < 2:
        return np.nan
    x = np.array([point[0] for point in points], dtype=float)
    x -= x[0]
    y = np.log1p(np.array([point[1] for point in points], dtype=float))
    return float(np.polyfit(x, y, 1)[0])


def _feature_row(
    values: dict[str, float],
    origin: str,
    horizon: int,
    destination: str,
) -> dict[str, float] | None:
    if origin not in values:
        return None
    last = values[origin]
    target_month = shift_month(origin, horizon)
    month_number = int(target_month[-2:])
    lag3 = values.get(shift_month(origin, -3), np.nan)
    lag6 = values.get(shift_month(origin, -6), np.nan)
    lag12 = values.get(shift_month(origin, -12), np.nan)
    row = {
        "destination": destination,
        "origin": origin,
        "horizon": horizon,
        "Date": target_month,
        "last_log": float(np.log1p(last)),
        "lag3_log": float(np.log1p(lag3)) if np.isfinite(lag3) else np.nan,
        "lag6_log": float(np.log1p(lag6)) if np.isfinite(lag6) else np.nan,
        "lag12_log": float(np.log1p(lag12)) if np.isfinite(lag12) else np.nan,
        "roll3_log": float(np.log1p(_rolling_mean(values, origin, 3))),
        "roll6_log": float(np.log1p(_rolling_mean(values, origin, 6))),
        "roll12_log": float(np.log1p(_rolling_mean(values, origin, 12))),
        "recent_change_3": float(np.log1p(last) - np.log1p(lag3))
        if np.isfinite(lag3)
        else np.nan,
        "recent_change_6": float(np.log1p(last) - np.log1p(lag6))
        if np.isfinite(lag6)
        else np.nan,
        "growth3": _slope(values, origin, 3),
        "growth6": _slope(values, origin, 6),
        "seasonal_rel_12": float(np.log1p(last) - np.log1p(lag12))
        if np.isfinite(lag12)
        else np.nan,
        "month_sin": float(np.sin(2 * np.pi * month_number / 12)),
        "month_cos": float(np.cos(2 * np.pi * month_number / 12)),
        "missing_lag3": float(not np.isfinite(lag3)),
        "missing_lag6": float(not np.isfinite(lag6)),
        "missing_lag12": float(not np.isfinite(lag12)),
        "_last_value": float(last),
    }
    return row


def add_destination_onehot(
    frame: pd.DataFrame, destinations: tuple[str, ...]
) -> pd.DataFrame:
    output = frame.copy()
    for destination in destinations:
        output[f"dest_{destination}"] = (output["destination"] == destination).astype(float)
    return output


def build_training_frame(
    history_wide: pd.DataFrame,
    destinations: tuple[str, ...],
    max_origin: str,
    max_horizon: int = 12,
    min_history_months: int = 12,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build supervised samples whose targets are not later than `max_origin`."""
    assert_no_future(history_wide, max_origin, "training-sample history")
    months = sorted(history_wide["Date"].unique(), key=month_ordinal)
    rows: list[dict[str, Any]] = []
    skipped_origins = 0
    for destination in destinations:
        values = _series_map(history_wide, destination)
        if not values:
            continue
        first_valid = min(values, key=month_ordinal)
        first_ordinal = month_ordinal(first_valid)
        for origin in months:
            if month_ordinal(origin) < first_ordinal + min_history_months:
                continue
            if month_ordinal(origin) > month_ordinal(max_origin):
                continue
            if origin not in values:
                skipped_origins += 1
                continue
            for horizon in range(1, max_horizon + 1):
                target_month = shift_month(origin, horizon)
                if month_ordinal(target_month) > month_ordinal(max_origin):
                    continue
                if target_month not in values:
                    continue
                row = _feature_row(values, origin, horizon, destination)
                if row is None:
                    continue
                last_log = row["last_log"]
                row["target_delta_log"] = float(
                    np.log1p(values[target_month]) - last_log
                )
                row["target_value"] = float(values[target_month])
                rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("no training samples were built")
    frame = add_destination_onehot(frame, destinations)
    info = {
        "n_samples": int(len(frame)),
        "n_origins": int(frame["origin"].nunique()),
        "n_destinations": int(frame["destination"].nunique()),
        "origin_range": [frame["origin"].min(), frame["origin"].max()],
        "skipped_origins": skipped_origins,
    }
    return frame, info


def build_prediction_frame(
    history_wide: pd.DataFrame,
    origin: str,
    forecast_months: tuple[str, ...],
    destinations: tuple[str, ...],
) -> pd.DataFrame:
    """Feature rows for one fixed forecast origin."""
    assert_no_future(history_wide, origin, "prediction features")
    history = history_wide.loc[history_wide["Date"] <= origin]
    rows: list[dict[str, Any]] = []
    for destination in destinations:
        values = _series_map(history, destination)
        if origin not in values:
            raise ValueError(f"{destination}: no observation at origin {origin}")
        for horizon, month in enumerate(forecast_months, start=1):
            if shift_month(origin, horizon) != month:
                raise ValueError("forecast months must be contiguous from the origin")
            row = _feature_row(values, origin, horizon, destination)
            if row is None:
                raise ValueError(f"{destination}: could not build features at {origin}")
            rows.append(row)
    frame = pd.DataFrame(rows)
    return add_destination_onehot(frame, destinations)


def _matrix_columns(destinations: tuple[str, ...]) -> list[str]:
    return [*FEATURE_COLUMNS, *[f"dest_{d}" for d in destinations]]


def _reconstruct(
    prediction_frame: pd.DataFrame,
    predicted_delta: np.ndarray,
    destinations: tuple[str, ...],
    model_label: str,
) -> pd.DataFrame:
    values = np.expm1(prediction_frame["last_log"].to_numpy(float) + predicted_delta)
    values = np.where(np.isfinite(values), values, prediction_frame["_last_value"].to_numpy(float))
    values = np.clip(values, 0.0, None)
    frame = prediction_frame[["Date", "destination", "horizon"]].copy()
    frame["forecast"] = values
    wide = frame.pivot(index="Date", columns="destination", values="forecast")
    wide = wide.loc[:, list(destinations)].reset_index()
    wide = wide[["Date", *destinations]]
    if not np.isfinite(wide[list(destinations)].to_numpy(float)).all():
        raise ValueError(f"{model_label}: non-finite reconstructed forecasts")
    return wide


def fit_ridge_frames(
    training: pd.DataFrame,
    prediction: pd.DataFrame,
    alpha: float,
    destinations: tuple[str, ...],
) -> pd.DataFrame:
    columns = _matrix_columns(destinations)
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=float(alpha))),
        ]
    )
    model.fit(training[columns], training["target_delta_log"])
    predicted_delta = model.predict(prediction[columns])
    return _reconstruct(prediction, predicted_delta, destinations, "global_ridge")


def fit_hgb_frames(
    training: pd.DataFrame,
    prediction: pd.DataFrame,
    params: dict[str, Any],
    destinations: tuple[str, ...],
    seed: int,
    min_samples_leaf: int = 20,
) -> pd.DataFrame:
    columns = _matrix_columns(destinations)
    model = HistGradientBoostingRegressor(
        **params,
        min_samples_leaf=int(min_samples_leaf),
        early_stopping=False,
        random_state=int(seed),
    )
    model.fit(training[columns], training["target_delta_log"])
    predicted_delta = model.predict(prediction[columns])
    return _reconstruct(prediction, predicted_delta, destinations, "global_hgb")


def fit_predict_ridge(
    history_wide: pd.DataFrame,
    cutoff: str,
    forecast_months: tuple[str, ...],
    alpha: float,
    destinations: tuple[str, ...],
    min_history_months: int = 12,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    history = history_wide.loc[history_wide["Date"] <= cutoff]
    training, train_info = build_training_frame(
        history, destinations, max_origin=cutoff,
        max_horizon=len(forecast_months), min_history_months=min_history_months,
    )
    prediction = build_prediction_frame(history, cutoff, forecast_months, destinations)
    wide = fit_ridge_frames(training, prediction, alpha, destinations)
    info = {
        "alpha": float(alpha),
        "n_features": len(_matrix_columns(destinations)),
        "training": train_info,
        "destination_columns": len(destinations),
    }
    return wide, info


def fit_predict_hgb(
    history_wide: pd.DataFrame,
    cutoff: str,
    forecast_months: tuple[str, ...],
    params: dict[str, Any],
    destinations: tuple[str, ...],
    seed: int,
    min_history_months: int = 12,
    min_samples_leaf: int = 20,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    history = history_wide.loc[history_wide["Date"] <= cutoff]
    training, train_info = build_training_frame(
        history, destinations, max_origin=cutoff,
        max_horizon=len(forecast_months), min_history_months=min_history_months,
    )
    prediction = build_prediction_frame(history, cutoff, forecast_months, destinations)
    wide = fit_hgb_frames(
        training, prediction, params, destinations, seed, min_samples_leaf
    )
    info = {
        "params": params,
        "min_samples_leaf": int(min_samples_leaf),
        "n_features": len(_matrix_columns(destinations)),
        "training": train_info,
    }
    return wide, info
