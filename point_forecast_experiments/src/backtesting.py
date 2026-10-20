"""Rolling-origin backtesting confined to dates <= 2022M07."""

from __future__ import annotations

import json
from typing import Any, Callable

import numpy as np
import pandas as pd

from .data_loader import TourismData, month_ordinal, shift_month
from .metrics import evaluate_point_forecast, forecast_errors_long
from .ml_models import (
    build_prediction_frame,
    build_training_frame,
    fit_hgb_frames,
    fit_ridge_frames,
)


def _max_horizon_for_origin(origin: str, train_end: str, max_horizon: int) -> int:
    return min(max_horizon, month_ordinal(train_end) - month_ordinal(origin))


def _backtest_months(origin: str, max_horizon: int) -> tuple[str, ...]:
    return tuple(shift_month(origin, h) for h in range(1, max_horizon + 1))


def _evaluate_origin(
    forecast: pd.DataFrame,
    actual: pd.DataFrame,
    history: pd.DataFrame,
    destinations: tuple[str, ...],
    model: str,
    params: dict[str, Any],
    origin: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    by_destination = evaluate_point_forecast(
        forecast, actual, history, destinations
    )
    errors = forecast_errors_long(
        forecast, actual, destinations, model=model, origin=origin
    )
    # Backtest targets can be missing (Chile/Maldives/Thailand pandemic gaps);
    # keep only pairs that the notebook metric would actually score.
    errors = errors.dropna(subset=["actual"]).copy()
    denominators = {}
    for destination in destinations:
        series = pd.to_numeric(
            history[destination], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan).dropna()
        denominators[destination] = series.diff().abs().dropna().mean()
    errors["mase"] = errors.apply(
        lambda row: row["ae"] / denominators[row["destination"]]
        if denominators.get(row["destination"], 0) > 0
        else np.nan,
        axis=1,
    )
    errors["params"] = json.dumps(params, sort_keys=True)
    errors["origin"] = origin
    summary = {
        "model": model,
        "params": json.dumps(params, sort_keys=True),
        "origin": origin,
        "mean_mae": float(by_destination["mae"].mean()),
        "mean_mase": float(by_destination["mase"].mean()),
        "median_mase": float(by_destination["mase"].median()),
        "mean_mape": float(by_destination["mape"].mean()),
        "n_destinations": int(len(by_destination)),
        "valid_samples": int(by_destination["n"].sum()),
        "horizon_max": int(errors["horizon"].max()),
    }
    return summary, errors


def backtest_statistical(
    model: str,
    param_sets: list[dict[str, Any]],
    forecast_fn: Callable[[pd.DataFrame, str, tuple[str, ...], dict[str, Any]], tuple[pd.DataFrame, dict[str, Any]]],
    data: TourismData,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    origins = config["backtest"]["origins"]
    max_horizon_limit = int(config["backtest"]["max_horizon"])
    train_end = config["split"]["train_end"]
    summaries: list[dict[str, Any]] = []
    details: list[pd.DataFrame] = []
    failures: list[dict[str, Any]] = []

    for params in param_sets:
        model_label = f"{model}({json.dumps(params, sort_keys=True)})"
        for origin in origins:
            max_h = _max_horizon_for_origin(origin, train_end, max_horizon_limit)
            if max_h <= 0:
                continue
            months = _backtest_months(origin, max_h)
            history = data.history.loc[data.history["Date"] <= origin].copy()
            actual = data.history.loc[data.history["Date"].isin(months)].copy()
            if len(actual) != len(months):
                failures.append(
                    {"model": model, "params": params, "origin": origin,
                     "error": "incomplete backtest actuals"}
                )
                continue
            try:
                forecast, info = forecast_fn(history, origin, months, params)
                summary, errors = _evaluate_origin(
                    forecast, actual, history, data.destinations,
                    model_label, params, origin,
                )
                summary["info"] = json.dumps(info, sort_keys=True, default=str)
                summaries.append(summary)
                details.append(errors)
            except Exception as exc:  # keep failures visible instead of fabricating
                failures.append(
                    {"model": model, "params": params, "origin": origin,
                     "error": f"{type(exc).__name__}: {exc}"}
                )
    summary_frame = pd.DataFrame(summaries)
    detail_frame = pd.concat(details, ignore_index=True) if details else pd.DataFrame()
    return summary_frame, detail_frame, failures


def backtest_global(
    model: str,
    kind: str,
    param_sets: list[dict[str, Any]],
    data: TourismData,
    config: dict[str, Any],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    origins = config["backtest"]["origins"]
    max_horizon_limit = int(config["backtest"]["max_horizon"])
    train_end = config["split"]["train_end"]
    summaries: list[dict[str, Any]] = []
    details: list[pd.DataFrame] = []
    failures: list[dict[str, Any]] = []

    for origin in origins:
        max_h = _max_horizon_for_origin(origin, train_end, max_horizon_limit)
        if max_h <= 0:
            continue
        months = _backtest_months(origin, max_h)
        history = data.history.loc[data.history["Date"] <= origin].copy()
        actual = data.history.loc[data.history["Date"].isin(months)].copy()
        if len(actual) != len(months):
            failures.append(
                {"model": model, "params": {}, "origin": origin,
                 "error": "incomplete backtest actuals"}
            )
            continue
        try:
            training, train_info = build_training_frame(
                history, data.destinations, max_origin=origin,
                max_horizon=max_h,
                min_history_months=int(
                    config["models"][
                        "global_ridge" if kind == "ridge" else "global_hgb"
                    ]["min_history_months"]
                ),
            )
            prediction = build_prediction_frame(
                history, origin, months, data.destinations
            )
        except Exception as exc:
            failures.append(
                {"model": model, "params": {}, "origin": origin,
                 "error": f"feature build failed: {type(exc).__name__}: {exc}"}
            )
            continue

        for params in param_sets:
            model_label = f"{model}({json.dumps(params, sort_keys=True)})"
            try:
                if kind == "ridge":
                    forecast = fit_ridge_frames(
                        training, prediction, float(params["alpha"]), data.destinations
                    )
                elif kind == "hgb":
                    forecast = fit_hgb_frames(
                        training, prediction, params, data.destinations, seed,
                        min_samples_leaf=int(
                            config["models"]["global_hgb"]["min_samples_leaf"]
                        ),
                    )
                else:
                    raise ValueError(f"unknown global model kind: {kind}")
                summary, errors = _evaluate_origin(
                    forecast, actual, history, data.destinations,
                    model_label, params, origin,
                )
                summary["info"] = json.dumps(
                    {"training": train_info}, sort_keys=True, default=str
                )
                summaries.append(summary)
                details.append(errors)
            except Exception as exc:
                failures.append(
                    {"model": model, "params": params, "origin": origin,
                     "error": f"{type(exc).__name__}: {exc}"}
                )
    summary_frame = pd.DataFrame(summaries)
    detail_frame = pd.concat(details, ignore_index=True) if details else pd.DataFrame()
    return summary_frame, detail_frame, failures


def aggregate_backtest(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    grouped = (
        summary.groupby(["model", "params"], as_index=False)
        .agg(
            n_origins=("origin", "nunique"),
            mean_mae=("mean_mae", "mean"),
            mean_mase=("mean_mase", "mean"),
            median_mase=("median_mase", "median"),
            mean_mape=("mean_mape", "mean"),
            total_valid_samples=("valid_samples", "sum"),
            mean_horizon_max=("horizon_max", "mean"),
        )
        .sort_values("mean_mase")
    )
    grouped["family"] = grouped["model"].str.split("(").str[0]
    return grouped
