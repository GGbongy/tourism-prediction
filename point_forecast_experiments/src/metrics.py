"""Metric definitions aligned with the starter notebook.

`evaluate_point_forecast` is a faithful copy of the notebook implementation:
MAE, MASE with a lag-1 scaling denominator taken from the supplied history,
and MAPE with zero actuals excluded.  Do not change these definitions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data_loader import month_ordinal


def evaluate_point_forecast(
    forecast_wide: pd.DataFrame,
    actual_wide: pd.DataFrame,
    scale_history_wide: pd.DataFrame,
    destinations: tuple[str, ...],
) -> pd.DataFrame:
    """Notebook-aligned per-destination MAE/MASE/MAPE."""
    forecast = forecast_wide.set_index("Date")
    actual = actual_wide.set_index("Date")
    rows = []
    for destination in destinations:
        paired = pd.concat(
            [
                actual[destination].rename("actual"),
                forecast[destination].rename("forecast"),
            ],
            axis=1,
            join="inner",
        ).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        errors = (paired["actual"] - paired["forecast"]).abs()
        training = pd.to_numeric(
            scale_history_wide[destination], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan).dropna()
        denominator = training.diff().abs().dropna().mean()
        nonzero = paired["actual"] != 0
        rows.append(
            {
                "destination": destination,
                "n": len(paired),
                "mae": errors.mean(),
                "mase": errors.mean() / denominator if denominator > 0 else np.nan,
                "mape": (errors[nonzero] / paired.loc[nonzero, "actual"].abs()).mean()
                * 100
                if nonzero.any()
                else np.nan,
                "mase_denominator": denominator,
                "mape_valid_n": int(nonzero.sum()),
            }
        )
    return pd.DataFrame(rows)


def forecast_errors_long(
    forecast_wide: pd.DataFrame,
    actual_wide: pd.DataFrame,
    destinations: tuple[str, ...],
    model: str,
    origin: str,
) -> pd.DataFrame:
    forecast = forecast_wide.melt(
        id_vars="Date", value_vars=list(destinations),
        var_name="destination", value_name="forecast",
    )
    actual = actual_wide.melt(
        id_vars="Date", value_vars=list(destinations),
        var_name="destination", value_name="actual",
    )
    merged = forecast.merge(actual, on=["Date", "destination"], how="inner")
    merged["model"] = model
    merged["horizon"] = merged["Date"].map(month_ordinal) - month_ordinal(origin)
    merged["ae"] = (merged["actual"] - merged["forecast"]).abs()
    merged["ape"] = np.where(
        merged["actual"] != 0,
        merged["ae"] / merged["actual"].abs() * 100,
        np.nan,
    )
    return merged


def per_horizon_metrics(
    errors_long: pd.DataFrame,
    scale_history_wide: pd.DataFrame,
    destinations: tuple[str, ...],
) -> pd.DataFrame:
    denominators = {}
    for destination in destinations:
        training = pd.to_numeric(
            scale_history_wide[destination], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan).dropna()
        denominators[destination] = training.diff().abs().dropna().mean()

    frame = errors_long.copy()
    frame["mase"] = frame.apply(
        lambda row: row["ae"] / denominators[row["destination"]]
        if denominators.get(row["destination"], 0) > 0
        else np.nan,
        axis=1,
    )
    grouped = (
        frame.groupby(["model", "horizon"], as_index=False)
        .agg(
            n=("ae", "size"),
            mean_mae=("ae", "mean"),
            mean_mase=("mase", "mean"),
            mean_mape=("ape", "mean"),
            mape_valid_n=("ape", "count"),
        )
        .sort_values(["model", "horizon"])
    )
    return grouped


def pooled_mae(errors_long: pd.DataFrame) -> float:
    return float(errors_long["ae"].mean())


def summary_row(
    model: str,
    by_destination: pd.DataFrame,
    errors_long: pd.DataFrame,
    runtime_seconds: float,
    note: str = "",
) -> dict[str, object]:
    return {
        "model": model,
        "MAE_macro": float(by_destination["mae"].mean()),
        "MAE_pooled": pooled_mae(errors_long),
        "mean_MASE": float(by_destination["mase"].mean()),
        "median_MASE": float(by_destination["mase"].median()),
        "mean_MAPE": float(by_destination["mape"].mean()),
        "valid_samples": int(by_destination["n"].sum()),
        "mape_valid_samples": int(by_destination["mape_valid_n"].sum()),
        "runtime_seconds": round(float(runtime_seconds), 4),
        "n_destinations": int(by_destination["destination"].nunique()),
        "note": note,
    }


def check_forecast_frame(
    forecast_wide: pd.DataFrame,
    expected_months: tuple[str, ...],
    destinations: tuple[str, ...],
    label: str,
) -> None:
    if tuple(forecast_wide.columns) != ("Date", *destinations):
        raise ValueError(f"{label}: wrong columns/order: {list(forecast_wide.columns)}")
    if tuple(forecast_wide["Date"]) != tuple(expected_months):
        raise ValueError(f"{label}: wrong Date rows: {forecast_wide['Date'].tolist()}")
    values = forecast_wide[list(destinations)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        bad = [
            (expected_months[i], destinations[j])
            for i, j in zip(*np.where(~np.isfinite(values)))
        ]
        raise ValueError(f"{label}: non-finite forecasts: {bad[:10]}")
    if (values < 0).any():
        bad = [
            (expected_months[i], destinations[j])
            for i, j in zip(*np.where(values < 0))
        ]
        raise ValueError(f"{label}: negative forecasts: {bad[:10]}")


def improvement_vs_baseline(
    by_destination: pd.DataFrame,
    baseline_model: str,
    candidate_models: list[str],
) -> pd.DataFrame:
    baseline = (
        by_destination.loc[by_destination["model"] == baseline_model]
        .set_index("destination")["mase"]
    )
    rows = []
    for model in candidate_models:
        candidate = (
            by_destination.loc[by_destination["model"] == model]
            .set_index("destination")["mase"]
        )
        joined = pd.concat([baseline.rename("baseline"), candidate.rename("model")], axis=1).dropna()
        if joined.empty:
            continue
        delta = joined["baseline"] - joined["model"]
        better = joined[delta > 0]
        worse = joined[delta < 0]
        rows.append(
            {
                "model": model,
                "baseline": baseline_model,
                "mean_mase_baseline": float(joined["baseline"].mean()),
                "mean_mase_model": float(joined["model"].mean()),
                "pct_improvement_mean_mase": float(
                    (joined["baseline"].mean() - joined["model"].mean())
                    / joined["baseline"].mean()
                    * 100
                ),
                "destinations_better": int(len(better)),
                "destinations_worse": int(len(worse)),
                "best_destination": better["baseline"].sub(better["model"]).idxmax()
                if len(better)
                else "",
                "worst_destination": worse["model"].sub(worse["baseline"]).idxmax()
                if len(worse)
                else "",
            }
        )
    return pd.DataFrame(rows)
