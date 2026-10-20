"""Run the full point-forecasting experiment suite.

Usage (from the repository root):
    .venv/Scripts/python.exe -m src.run_experiments

The script writes every table described in the task to results/ and never
touches the locked audit or final-period actuals.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .backtesting import (
    aggregate_backtest,
    backtest_global,
    backtest_statistical,
)
from .baselines import (
    moving_average_forecast,
    naive_lag1,
    naive_lag12,
    naive_lag12_raw,
)
from .data_loader import TourismData, load_data, month_ordinal
from .ensemble import blend_forecasts
from .metrics import (
    check_forecast_frame,
    evaluate_point_forecast,
    forecast_errors_long,
    improvement_vs_baseline,
    per_horizon_metrics,
    summary_row,
)
from .ml_models import fit_predict_hgb, fit_predict_ridge
from .statistical_models import (
    adaptive_seasonal_forecast,
    damped_trend_forecast,
)

EXP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXP_DIR.parent

EXPECTED_NOTEBOOK_BASELINES = {
    "naive_lag1": 0.866913,
    "naive_lag12": 1.413921,
}


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def _damped_label(params: dict[str, Any]) -> str:
    return (
        f"damped_trend_w{params['window']}_phi{params['damping']}"
        f"_cap{params['trend_cap']}"
    )


def _adaptive_label(params: dict[str, Any]) -> str:
    low, high = params["ratio_cap"]
    return f"adaptive_seasonal_w{params['window']}_cap{low}-{high}"


def _ridge_label(params: dict[str, Any]) -> str:
    return f"global_ridge_a{params['alpha']}"


def _hgb_label(params: dict[str, Any]) -> str:
    return (
        f"global_hgb_i{params['max_iter']}_l{params['max_leaf_nodes']}"
        f"_l2{params['l2_regularization']}"
    )


def _select_top(summary: pd.DataFrame, top_k: int) -> list[dict[str, Any]]:
    if summary.empty:
        return []
    aggregated = aggregate_backtest(summary)
    selected = []
    for raw in aggregated.head(top_k)["params"]:
        selected.append(json.loads(raw))
    return selected


def _run_validation(
    model_name: str,
    forecast_fn,
    params: dict[str, Any],
    data: TourismData,
    origin: str,
    months: tuple[str, ...],
) -> dict[str, Any]:
    start = time.perf_counter()
    forecast, info = forecast_fn(data.history, origin, months, params)
    runtime = time.perf_counter() - start
    check_forecast_frame(forecast, months, data.destinations, model_name)
    by_destination = evaluate_point_forecast(
        forecast, data.validation_actuals, data.history, data.destinations
    )
    by_destination.insert(0, "model", model_name)
    errors = forecast_errors_long(
        forecast, data.validation_actuals, data.destinations, model_name, origin
    )
    summary = summary_row(model_name, by_destination, errors, runtime)
    summary["params"] = json.dumps(params, sort_keys=True)
    summary["info"] = json.dumps(info, sort_keys=True, default=str)
    return {
        "model": model_name,
        "forecast": forecast,
        "info": info,
        "by_destination": by_destination,
        "errors": errors,
        "summary": summary,
        "runtime_seconds": runtime,
    }


def _forecast_damped(history, cutoff, months, params, config, destinations):
    model_config = config["models"]["damped_trend"]
    return damped_trend_forecast(
        history_wide=history,
        cutoff=cutoff,
        forecast_months=months,
        window=int(params["window"]),
        damping=float(params["damping"]),
        trend_cap=float(params["trend_cap"]),
        max_total_log_growth=model_config.get("max_total_log_growth"),
        min_points=int(model_config["min_points"]),
        destinations=destinations,
    )


def _forecast_adaptive(history, cutoff, months, params, config, destinations):
    return adaptive_seasonal_forecast(
        history_wide=history,
        cutoff=cutoff,
        forecast_months=months,
        window=int(params["window"]),
        ratio_cap=tuple(params["ratio_cap"]),
        min_pairs=int(config["models"]["adaptive_seasonal"]["min_pairs"]),
        fallback_years=int(
            config["models"]["adaptive_seasonal"]["fallback_years"]
        ),
        destinations=destinations,
    )


def _forecast_ridge(history, cutoff, months, params, config, destinations):
    return fit_predict_ridge(
        history_wide=history,
        cutoff=cutoff,
        forecast_months=months,
        alpha=float(params["alpha"]),
        destinations=destinations,
        min_history_months=int(
            config["models"]["global_ridge"]["min_history_months"]
        ),
    )


def _forecast_hgb(history, cutoff, months, params, config, destinations, seed):
    return fit_predict_hgb(
        history_wide=history,
        cutoff=cutoff,
        forecast_months=months,
        params=params,
        destinations=destinations,
        seed=seed,
        min_history_months=int(
            config["models"]["global_hgb"]["min_history_months"]
        ),
        min_samples_leaf=int(config["models"]["global_hgb"]["min_samples_leaf"]),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=EXP_DIR / "configs" / "experiment_config.json",
    )
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)

    results_dir = EXP_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (EXP_DIR / "figures").mkdir(parents=True, exist_ok=True)

    experiment_start = time.perf_counter()
    data = load_data(config, PROJECT_ROOT)
    origin = config["validation_origin"]
    months = data.validation_months
    destinations = data.destinations
    failures: list[dict[str, Any]] = []
    log: dict[str, Any] = {
        "started_at": pd.Timestamp.now().isoformat(),
        "seed": seed,
        "python": sys.version,
        "platform": platform.platform(),
        "environment": {
            package: version(package)
            for package in (
                "numpy", "pandas", "scikit-learn", "scipy",
                "matplotlib", "jupyterlab", "nbformat",
            )
        },
        "config": config,
        "data": data.metadata,
        "audit_used": False,
        "audit_isolation": {
            "history_max_date": data.history["Date"].max(),
            "validation_months": list(months),
            "note": "locked audit and final-period actuals are not loaded into the experiment object",
        },
        "failures": failures,
    }

    # --- Notebook baseline consistency check (raw, no fallback) -------------
    print("Checking notebook baselines...")
    raw_lag1, _ = naive_lag1(data.history, origin, months, destinations)
    raw_lag12 = naive_lag12_raw(data.history, origin, months, destinations)
    raw_lag1_metrics = evaluate_point_forecast(
        raw_lag1, data.validation_actuals, data.history, destinations
    )
    raw_lag12_metrics = evaluate_point_forecast(
        raw_lag12, data.validation_actuals, data.history, destinations
    )
    baseline_check = {}
    for name, metrics in (("naive_lag1", raw_lag1_metrics), ("naive_lag12", raw_lag12_metrics)):
        actual_mean = float(metrics["mase"].mean())
        expected = EXPECTED_NOTEBOOK_BASELINES[name]
        baseline_check[name] = {
            "computed_mean_mase": actual_mean,
            "notebook_mean_mase": expected,
            "abs_diff": abs(actual_mean - expected),
            "matches": bool(abs(actual_mean - expected) < 1e-5),
            "valid_samples": int(metrics["n"].sum()),
        }
    log["notebook_baseline_check"] = baseline_check
    if not all(item["matches"] for item in baseline_check.values()):
        failures.append(
            {"model": "baseline_check", "error": "notebook baseline mismatch",
             "detail": baseline_check}
        )
    print(json.dumps(baseline_check, indent=2, default=_json_default))

    # --- Backtesting: narrow hyperparameters --------------------------------
    print("Running rolling-origin backtests...")
    backtest_details: list[pd.DataFrame] = []
    backtest_summaries: list[pd.DataFrame] = []
    backtest_failures: list[dict[str, Any]] = []
    backtest_runtime: dict[str, float] = {}

    def record_backtest(name: str, summary: pd.DataFrame, details: pd.DataFrame, fails: list[dict[str, Any]], elapsed: float):
        backtest_runtime[name] = elapsed
        if not summary.empty:
            summary = summary.copy()
            summary["family"] = name
            backtest_summaries.append(summary)
        if not details.empty:
            details = details.copy()
            details["family"] = name
            backtest_details.append(details)
        backtest_failures.extend(fails)

    # Simple models have no hyperparameters.
    simple_specs = {
        "naive_lag1": lambda history, cutoff, forecast_months, params: naive_lag1(
            history, cutoff, forecast_months, destinations
        ),
        "naive_lag12": lambda history, cutoff, forecast_months, params: naive_lag12(
            history, cutoff, forecast_months, destinations
        ),
        "recent_ma_3": lambda history, cutoff, forecast_months, params: moving_average_forecast(
            history, cutoff, forecast_months, 3, destinations
        ),
        "recent_ma_6": lambda history, cutoff, forecast_months, params: moving_average_forecast(
            history, cutoff, forecast_months, 6, destinations
        ),
    }
    for name, fn in simple_specs.items():
        start = time.perf_counter()
        summary, details, fails = backtest_statistical(
            name, [{}], fn, data, config
        )
        record_backtest(name, summary, details, fails, time.perf_counter() - start)
        print(f"  backtest {name}: {len(summary)} origin rows")

    # Damped trend grid.
    damped_config = config["models"]["damped_trend"]
    damped_grid = [
        {"window": window, "damping": damping, "trend_cap": cap}
        for window in damped_config["windows"]
        for damping in damped_config["dampings"]
        for cap in damped_config["trend_caps"]
    ]
    damped_fn = lambda history, cutoff, forecast_months, params: _forecast_damped(
        history, cutoff, forecast_months, params, config, destinations
    )
    start = time.perf_counter()
    damped_summary, damped_details, damped_fails = backtest_statistical(
        "damped_trend", damped_grid, damped_fn, data, config
    )
    record_backtest(
        "damped_trend", damped_summary, damped_details, damped_fails,
        time.perf_counter() - start,
    )
    print(f"  backtest damped_trend: {len(damped_summary)} origin rows")

    # Adaptive seasonal grid.
    adaptive_config = config["models"]["adaptive_seasonal"]
    adaptive_grid = [
        {"window": window, "ratio_cap": list(caps)}
        for window in adaptive_config["windows"]
        for caps in adaptive_config["ratio_caps"]
    ]
    adaptive_fn = lambda history, cutoff, forecast_months, params: _forecast_adaptive(
        history, cutoff, forecast_months, params, config, destinations
    )
    start = time.perf_counter()
    adaptive_summary, adaptive_details, adaptive_fails = backtest_statistical(
        "adaptive_seasonal", adaptive_grid, adaptive_fn, data, config
    )
    record_backtest(
        "adaptive_seasonal", adaptive_summary, adaptive_details, adaptive_fails,
        time.perf_counter() - start,
    )
    print(f"  backtest adaptive_seasonal: {len(adaptive_summary)} origin rows")

    # Global Ridge grid.
    ridge_grid = [
        {"alpha": alpha} for alpha in config["models"]["global_ridge"]["alphas"]
    ]
    start = time.perf_counter()
    ridge_summary, ridge_details, ridge_fails = backtest_global(
        "global_ridge", "ridge", ridge_grid, data, config, seed
    )
    record_backtest(
        "global_ridge", ridge_summary, ridge_details, ridge_fails,
        time.perf_counter() - start,
    )
    print(f"  backtest global_ridge: {len(ridge_summary)} origin rows")

    # Global HistGradientBoosting grid.
    hgb_grid = [dict(params) for params in config["models"]["global_hgb"]["param_sets"]]
    start = time.perf_counter()
    hgb_summary, hgb_details, hgb_fails = backtest_global(
        "global_hgb", "hgb", hgb_grid, data, config, seed
    )
    record_backtest(
        "global_hgb", hgb_summary, hgb_details, hgb_fails,
        time.perf_counter() - start,
    )
    print(f"  backtest global_hgb: {len(hgb_summary)} origin rows")

    backtest_failures_all = backtest_failures
    failures.extend(backtest_failures_all)

    # --- Select candidates from backtest, then validate ---------------------
    top_k = int(config["selection"]["top_backtest_candidates"])
    selected = {
        "damped_trend": _select_top(damped_summary, top_k),
        "adaptive_seasonal": _select_top(adaptive_summary, top_k),
        "global_ridge": _select_top(ridge_summary, top_k),
        "global_hgb": _select_top(hgb_summary, top_k),
    }
    log["selected_candidates_by_backtest"] = selected
    print("Selected by backtest:", json.dumps(selected, indent=2, default=_json_default))

    print("Running validation forecasts...")
    validation_records: list[dict[str, Any]] = []

    # Simple models.
    simple_validation = [
        ("naive_lag1", lambda h, c, m, p: naive_lag1(h, c, m, destinations), {}),
        ("naive_lag12", lambda h, c, m, p: naive_lag12(h, c, m, destinations), {}),
        ("recent_ma_3", lambda h, c, m, p: moving_average_forecast(h, c, m, 3, destinations), {}),
        ("recent_ma_6", lambda h, c, m, p: moving_average_forecast(h, c, m, 6, destinations), {}),
    ]
    for name, fn, params in simple_validation:
        try:
            record = _run_validation(name, fn, params, data, origin, months)
            validation_records.append(record)
            print(f"  {name}: mean MASE={record['summary']['mean_MASE']:.4f}")
        except Exception as exc:
            failures.append({"model": name, "error": f"{type(exc).__name__}: {exc}"})

    # Parameterised families, only the backtest-selected candidates.
    for params in selected["damped_trend"]:
        name = _damped_label(params)
        try:
            record = _run_validation(
                name,
                lambda h, c, m, p: _forecast_damped(h, c, m, p, config, destinations),
                params, data, origin, months,
            )
            validation_records.append(record)
            print(f"  {name}: mean MASE={record['summary']['mean_MASE']:.4f}")
        except Exception as exc:
            failures.append({"model": name, "error": f"{type(exc).__name__}: {exc}"})

    for params in selected["adaptive_seasonal"]:
        name = _adaptive_label(params)
        try:
            record = _run_validation(
                name,
                lambda h, c, m, p: _forecast_adaptive(h, c, m, p, config, destinations),
                params, data, origin, months,
            )
            validation_records.append(record)
            print(f"  {name}: mean MASE={record['summary']['mean_MASE']:.4f}")
        except Exception as exc:
            failures.append({"model": name, "error": f"{type(exc).__name__}: {exc}"})

    for params in selected["global_ridge"]:
        name = _ridge_label(params)
        try:
            record = _run_validation(
                name,
                lambda h, c, m, p: _forecast_ridge(h, c, m, p, config, destinations),
                params, data, origin, months,
            )
            validation_records.append(record)
            print(f"  {name}: mean MASE={record['summary']['mean_MASE']:.4f}")
        except Exception as exc:
            failures.append({"model": name, "error": f"{type(exc).__name__}: {exc}"})

    for params in selected["global_hgb"]:
        name = _hgb_label(params)
        try:
            record = _run_validation(
                name,
                lambda h, c, m, p: _forecast_hgb(h, c, m, p, config, destinations, seed),
                params, data, origin, months,
            )
            validation_records.append(record)
            print(f"  {name}: mean MASE={record['summary']['mean_MASE']:.4f}")
        except Exception as exc:
            failures.append({"model": name, "error": f"{type(exc).__name__}: {exc}"})

    if not validation_records:
        raise RuntimeError("no validation model executed successfully")

    # --- Ensembles ----------------------------------------------------------
    print("Building weighted ensembles...")
    by_model = {record["model"]: record for record in validation_records}
    family_best: dict[str, str] = {}
    for family in ("adaptive_seasonal", "damped_trend", "global_ridge", "global_hgb"):
        candidates = [
            record["model"] for record in validation_records
            if record["model"].startswith(f"{family}")
        ]
        if candidates:
            best = min(
                candidates,
                key=lambda name: by_model[name]["summary"]["mean_MASE"],
            )
            family_best[family] = best
    log["family_best_validation"] = family_best

    ensemble_config = config["models"]["ensemble"]
    member_names = [
        family_best[key]
        for key in ("adaptive_seasonal", "damped_trend", "global_ridge", "global_hgb")
        if key in family_best
    ]
    ranked_members = sorted(
        member_names, key=lambda name: by_model[name]["summary"]["mean_MASE"]
    )
    ensemble_records: list[dict[str, Any]] = []
    ensemble_search_rows: list[dict[str, Any]] = []

    def make_ensemble(members: list[str], weights: list[float]) -> None:
        label = "ensemble_" + "+".join(members) + "_w" + "_".join(
            f"{weight:.2f}" for weight in weights
        )
        start = time.perf_counter()
        forecast, info = blend_forecasts(
            {name: by_model[name]["forecast"] for name in members},
            members, weights, destinations, months,
        )
        runtime = time.perf_counter() - start
        check_forecast_frame(forecast, months, destinations, label)
        by_destination = evaluate_point_forecast(
            forecast, data.validation_actuals, data.history, destinations
        )
        by_destination.insert(0, "model", label)
        errors = forecast_errors_long(
            forecast, data.validation_actuals, destinations, label, origin
        )
        summary = summary_row(label, by_destination, errors, runtime)
        summary["params"] = json.dumps(info, sort_keys=True)
        summary["info"] = json.dumps(info, sort_keys=True)
        record = {
            "model": label,
            "forecast": forecast,
            "info": info,
            "by_destination": by_destination,
            "errors": errors,
            "summary": summary,
            "runtime_seconds": runtime,
        }
        ensemble_records.append(record)
        ensemble_search_rows.append(
            {
                "model": label,
                "members": "|".join(members),
                "weights": "|".join(f"{weight:.6f}" for weight in weights),
                "mean_MASE": summary["mean_MASE"],
                "median_MASE": summary["median_MASE"],
                "mean_MAPE": summary["mean_MAPE"],
                "MAE_macro": summary["MAE_macro"],
            }
        )
        print(f"  {label}: mean MASE={summary['mean_MASE']:.4f}")

    if len(ranked_members) >= 2:
        top2 = ranked_members[:2]
        for weights in ensemble_config["weight_sets_2"]:
            make_ensemble(top2, list(weights))
    if len(ranked_members) >= 3:
        top3 = ranked_members[:3]
        for weights in ensemble_config["weight_sets_3"]:
            make_ensemble(top3, list(weights))
    if len(ranked_members) >= 4:
        make_ensemble(ranked_members[:4], [0.25, 0.25, 0.25, 0.25])

    all_records = validation_records + ensemble_records

    # --- Assemble result tables --------------------------------------------
    metrics_summary = pd.DataFrame([record["summary"] for record in all_records])
    metrics_summary = metrics_summary.sort_values("mean_MASE").reset_index(drop=True)
    metrics_by_destination = pd.concat(
        [record["by_destination"] for record in all_records], ignore_index=True
    )
    validation_predictions = pd.concat(
        [record["errors"] for record in all_records], ignore_index=True
    )
    horizon_frames = []
    for record in all_records:
        frame = per_horizon_metrics(
            record["errors"], data.history, destinations
        )
        horizon_frames.append(frame)
    metrics_by_horizon = pd.concat(horizon_frames, ignore_index=True)

    backtest_metrics = (
        pd.concat(backtest_details, ignore_index=True)
        if backtest_details
        else pd.DataFrame()
    )
    backtest_summary = (
        pd.concat(backtest_summaries, ignore_index=True)
        if backtest_summaries
        else pd.DataFrame()
    )
    backtest_summary["params"] = backtest_summary["params"].astype(str)
    backtest_aggregate = aggregate_backtest(backtest_summary)

    runtime_rows = []
    for record in all_records:
        runtime_rows.append(
            {
                "model": record["model"],
                "validation_runtime_seconds": round(record["runtime_seconds"], 4),
                "family": "ensemble"
                if any(record is item for item in ensemble_records)
                else "",
            }
        )
    runtime_frame = pd.DataFrame(runtime_rows)
    runtime_frame["note"] = "validation fit+predict time only; backtest totals are in experiment_log.json"

    candidates = [name for name in metrics_summary["model"] if name not in {"naive_lag1", "naive_lag12"}]
    improvement_lag1 = improvement_vs_baseline(
        metrics_by_destination, "naive_lag1", candidates
    )
    improvement_lag12 = improvement_vs_baseline(
        metrics_by_destination, "naive_lag12", candidates
    )

    # --- Integrity checks ---------------------------------------------------
    for record in all_records:
        check_forecast_frame(
            record["forecast"], months, destinations, record["model"]
        )
    if validation_predictions[["actual", "forecast"]].isna().any().any():
        raise AssertionError("validation predictions contain missing values")
    if not np.isfinite(
        validation_predictions[["actual", "forecast", "ae"]].to_numpy(float)
    ).all():
        raise AssertionError("validation predictions contain non-finite values")
    if (validation_predictions["forecast"] < 0).any():
        raise AssertionError("negative forecasts found")

    # Cross-check summary and destination tables.
    recomputed = (
        metrics_by_destination.groupby("model", as_index=False)
        .agg(mean_mase=("mase", "mean"), mean_mae=("mae", "mean"))
    )
    merged = metrics_summary.merge(recomputed, on="model", suffixes=("", "_recomputed"))
    if not np.allclose(merged["mean_MASE"], merged["mean_mase"]):
        raise AssertionError("mean_MASE is inconsistent between summary and destination tables")

    # --- Save outputs -------------------------------------------------------
    metrics_summary.to_csv(results_dir / "metrics_summary.csv", index=False)
    metrics_by_destination.to_csv(
        results_dir / "metrics_by_destination.csv", index=False
    )
    metrics_by_horizon.to_csv(results_dir / "metrics_by_horizon.csv", index=False)
    validation_predictions.to_csv(
        results_dir / "validation_predictions.csv", index=False
    )
    backtest_metrics.to_csv(results_dir / "backtest_metrics.csv", index=False)
    backtest_aggregate.to_csv(results_dir / "backtest_summary.csv", index=False)
    runtime_frame.to_csv(results_dir / "runtime_comparison.csv", index=False)
    pd.DataFrame(ensemble_search_rows).to_csv(
        results_dir / "ensemble_search.csv", index=False
    )
    pd.concat(
        [
            improvement_lag1.assign(baseline="naive_lag1"),
            improvement_lag12.assign(baseline="naive_lag12"),
        ],
        ignore_index=True,
    ).to_csv(results_dir / "improvement_vs_baselines.csv", index=False)

    total_runtime = time.perf_counter() - experiment_start
    best = metrics_summary.iloc[0].to_dict()
    log.update(
        {
            "finished_at": pd.Timestamp.now().isoformat(),
            "total_runtime_seconds": round(total_runtime, 3),
            "backtest_runtime_seconds": {
                key: round(value, 3) for key, value in backtest_runtime.items()
            },
            "backtest_failures": backtest_failures_all,
            "validation_failures": [
                item for item in failures if item.get("model") not in backtest_runtime
            ],
            "models_completed": list(metrics_summary["model"]),
            "best_model": best["model"],
            "best_model_metrics": {
                key: best[key]
                for key in (
                    "MAE_macro", "MAE_pooled", "mean_MASE", "median_MASE",
                    "mean_MAPE", "valid_samples", "runtime_seconds",
                )
            },
            "selected_candidates_by_backtest": selected,
            "backtest_aggregate_top": backtest_aggregate.head(10).to_dict(
                orient="records"
            ),
        }
    )
    _write_json(results_dir / "experiment_log.json", log)

    print("\nTop models by validation mean MASE:")
    print(
        metrics_summary[
            ["model", "MAE_macro", "mean_MASE", "median_MASE", "mean_MAPE", "runtime_seconds"]
        ].head(15).to_string(index=False)
    )
    print(f"\nSaved results to {results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
