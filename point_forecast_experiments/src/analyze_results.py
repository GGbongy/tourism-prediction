"""Post-experiment analysis used by the report.

Produces per-origin / per-horizon backtest tables, relative-to-lag-1
stability tables, and backtest reconstructions of the candidate ensembles.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EXP_DIR = Path(__file__).resolve().parents[1]
RESULTS = EXP_DIR / "results"


def _parse_validation_name(name: str) -> dict[str, Any]:
    if name.startswith("global_ridge"):
        return {"alpha": float(re.search(r"_a([0-9.]+)$", name).group(1))}
    if name.startswith("damped_trend"):
        match = re.search(r"_w(\d+)_phi([0-9.]+)_cap([0-9.]+)$", name)
        return {
            "window": int(match.group(1)),
            "damping": float(match.group(2)),
            "trend_cap": float(match.group(3)),
        }
    if name.startswith("global_hgb"):
        match = re.search(r"_i(\d+)_l(\d+)_l2([0-9.]+)$", name)
        return {
            "max_iter": int(match.group(1)),
            "max_leaf_nodes": int(match.group(2)),
            "l2_regularization": float(match.group(3)),
            "learning_rate": 0.1,
        }
    if name.startswith("adaptive_seasonal"):
        match = re.search(r"_w(\d+)_cap([0-9.]+)-([0-9.]+)$", name)
        return {
            "window": int(match.group(1)),
            "ratio_cap": [float(match.group(2)), float(match.group(3))],
        }
    raise ValueError(f"cannot parse model label: {name}")


def _backtest_label(family: str, params: dict[str, Any]) -> str:
    return f"{family}({json.dumps(params, sort_keys=True)})"


def load_backtest() -> pd.DataFrame:
    frame = pd.read_csv(RESULTS / "backtest_metrics.csv")
    frame["family"] = frame["model"].str.split("(").str[0]
    return frame


def build_ensemble_backtest(
    frame: pd.DataFrame,
    members: dict[str, float],
    name: str,
) -> pd.DataFrame:
    key = ["origin", "Date", "destination", "horizon"]
    forecasts = []
    for label in members:
        sub = frame.loc[frame["model"] == label, key + ["forecast"]]
        if sub.empty:
            raise ValueError(f"ensemble member missing from backtest: {label}")
        forecasts.append(
            sub.set_index(key)[["forecast"]].rename(columns={"forecast": label})
        )
    blended = pd.concat(forecasts, axis=1).reset_index()
    blended["forecast"] = sum(
        blended[label] * weight for label, weight in members.items()
    )
    actual = frame.loc[frame["model"] == next(iter(members)), key + ["actual"]]
    blended = blended[key + ["forecast"]].merge(actual, on=key, how="left")
    baseline = frame.loc[
        frame["model"] == "naive_lag1({})", key + ["ae", "mase"]
    ].rename(columns={"ae": "ae_lag1", "mase": "mase_lag1"})
    blended = blended.merge(baseline, on=key, how="left")
    blended["denominator"] = blended["ae_lag1"] / blended["mase_lag1"]
    blended["ae"] = (blended["actual"] - blended["forecast"]).abs()
    blended["ape"] = np.where(
        blended["actual"] != 0,
        blended["ae"] / blended["actual"].abs() * 100,
        np.nan,
    )
    blended["mase"] = blended["ae"] / blended["denominator"]
    blended["model"] = name
    return blended


def _short_labels() -> dict[str, str]:
    summary = pd.read_csv(RESULTS / "metrics_summary.csv")
    models = list(summary["model"])
    e4 = next(m for m in models if "adaptive_seasonal" in m)
    e3 = next(
        m for m in models
        if "global_hgb" in m and "adaptive" not in m and "w0.33_0.33_0.33" in m
    )
    e2_75 = next(
        m for m in models
        if "global_hgb" not in m and "adaptive" not in m and "w0.75_0.25" in m
    )
    e2_50 = next(
        m for m in models
        if "global_hgb" not in m and "adaptive" not in m and "w0.50_0.50" in m
    )
    return {
        "Lag1": "naive_lag1",
        "Lag12": "naive_lag12",
        "MA3": "recent_ma_3",
        "MA6": "recent_ma_6",
        "Damped": "damped_trend_w12_phi0.7_cap0.15",
        "Adaptive": "adaptive_seasonal_w3_cap0.5-3.0",
        "Ridge": "global_ridge_a0.1",
        "HGB": "global_hgb_i100_l15_l21.0",
        "E2-75/25": e2_75,
        "E2-50/50": e2_50,
        "E3-equal": e3,
        "E4-equal": e4,
    }


def main() -> int:
    log = json.loads((RESULTS / "experiment_log.json").read_text(encoding="utf-8"))
    frame = load_backtest()
    labels = _short_labels()
    label_to_short = {value: key for key, value in labels.items()}

    # Raw and relative-to-lag1 backtest tables.
    key = ["origin", "destination", "horizon"]
    baseline = (
        frame.loc[frame["model"] == "naive_lag1({})", key + ["mase"]]
        .rename(columns={"mase": "mase_lag1"})
    )
    relative = frame.merge(baseline, on=key, how="left")
    relative["relative_mase"] = relative["mase"] / relative["mase_lag1"]

    by_origin = (
        frame.groupby(["family", "model", "origin"], as_index=False)
        .agg(
            mean_mase=("mase", "mean"),
            median_mase=("mase", "median"),
            mean_mae=("ae", "mean"),
            mean_mape=("ape", "mean"),
            n=("mase", "size"),
        )
    )
    by_origin.to_csv(RESULTS / "backtest_by_origin.csv", index=False)
    by_horizon = (
        frame.groupby(["family", "model", "horizon"], as_index=False)
        .agg(
            mean_mase=("mase", "mean"),
            mean_mae=("ae", "mean"),
            mean_mape=("ape", "mean"),
            n=("mase", "size"),
        )
    )
    by_horizon.to_csv(RESULTS / "backtest_by_horizon.csv", index=False)
    relative.groupby(["family", "model", "origin"], as_index=False).agg(
        relative_mase=("relative_mase", "mean"), n=("relative_mase", "size")
    ).to_csv(RESULTS / "backtest_relative_by_origin.csv", index=False)
    relative.groupby(["family", "model", "horizon"], as_index=False).agg(
        relative_mase=("relative_mase", "mean"), n=("relative_mase", "size")
    ).to_csv(RESULTS / "backtest_relative_by_horizon.csv", index=False)

    # Backtest reconstructions of the candidate ensembles.
    family_best = log["family_best_validation"]
    member_labels = {
        family: _backtest_label(family, _parse_validation_name(name))
        for family, name in family_best.items()
    }
    variants = {
        "E4-equal": {
            member_labels["global_ridge"]: 0.25,
            member_labels["damped_trend"]: 0.25,
            member_labels["global_hgb"]: 0.25,
            member_labels["adaptive_seasonal"]: 0.25,
        },
        "E3-equal": {
            member_labels["global_ridge"]: 1 / 3,
            member_labels["damped_trend"]: 1 / 3,
            member_labels["global_hgb"]: 1 / 3,
        },
        "E2-50/50": {
            member_labels["global_ridge"]: 0.5,
            member_labels["damped_trend"]: 0.5,
        },
        "E2-75/25": {
            member_labels["global_ridge"]: 0.75,
            member_labels["damped_trend"]: 0.25,
        },
    }
    ensemble_frames = [
        build_ensemble_backtest(frame, members, name)
        for name, members in variants.items()
    ]
    ensemble_all = pd.concat(ensemble_frames, ignore_index=True)
    ensemble_all.to_csv(RESULTS / "backtest_ensemble_variants.csv", index=False)
    ensemble_by_origin = (
        ensemble_all.groupby(["model", "origin"], as_index=False)
        .agg(mean_mase=("mase", "mean"), mean_mae=("ae", "mean"), mean_mape=("ape", "mean"))
    )
    ensemble_by_horizon = (
        ensemble_all.groupby(["model", "horizon"], as_index=False)
        .agg(mean_mase=("mase", "mean"), mean_mae=("ae", "mean"), mean_mape=("ape", "mean"))
    )
    ensemble_by_origin.to_csv(RESULTS / "backtest_ensemble_by_origin.csv", index=False)
    ensemble_by_horizon.to_csv(RESULTS / "backtest_ensemble_by_horizon.csv", index=False)
    recovery_origins = ["2020M12", "2021M03"]
    ensemble_recovery = (
        ensemble_all.loc[ensemble_all["origin"].isin(recovery_origins)]
        .groupby(["model", "horizon"], as_index=False)
        .agg(mean_mase=("mase", "mean"), mean_mae=("ae", "mean"), mean_mape=("ape", "mean"))
    )
    ensemble_recovery.to_csv(
        RESULTS / "backtest_ensemble_recovery_by_horizon.csv", index=False
    )

    # Report facts.
    summary = pd.read_csv(RESULTS / "metrics_summary.csv")
    by_destination = pd.read_csv(RESULTS / "metrics_by_destination.csv")
    by_horizon = pd.read_csv(RESULTS / "metrics_by_horizon.csv")
    predictions = pd.read_csv(RESULTS / "validation_predictions.csv")
    lag1_destination = (
        by_destination.loc[by_destination["model"] == "naive_lag1"]
        .set_index("destination")["mase"]
        .sort_index()
    )

    key_model_rows = []
    for short, model in labels.items():
        model_summary = summary.loc[summary["model"] == model]
        destination = by_destination.loc[by_destination["model"] == model].set_index("destination")
        predictions_model = predictions.loc[predictions["model"] == model]
        if model_summary.empty or destination.empty:
            continue
        row = model_summary.iloc[0].to_dict()
        row["short"] = short
        row["destinations_better_than_lag1"] = int(
            (destination["mase"].sort_index() < lag1_destination).sum()
        )
        row["destinations_worse_than_lag1"] = int(
            (destination["mase"].sort_index() > lag1_destination).sum()
        )
        row["worst_destination"] = destination["mase"].idxmax()
        row["worst_destination_mase"] = float(destination["mase"].max())
        row["best_destination"] = destination["mase"].idxmin()
        row["best_destination_mase"] = float(destination["mase"].min())
        row["max_abs_error"] = float(predictions_model["ae"].max())
        key_model_rows.append(row)
    key_models = pd.DataFrame(key_model_rows).sort_values("mean_MASE")

    destination_detail = {}
    for short in ("MA3", "E2-75/25", "E3-equal", "E4-equal", "Ridge"):
        model = labels[short]
        detail = (
            by_destination.loc[by_destination["model"] == model]
            .sort_values("mase", ascending=False)
        )
        destination_detail[short] = {
            "worst": detail.head(5)[
                ["destination", "mae", "mase", "mape", "n"]
            ].to_dict(orient="records"),
            "best": detail.tail(5)[
                ["destination", "mae", "mase", "mape", "n"]
            ].to_dict(orient="records"),
        }

    failure_cases = {}
    for short in ("MA3", "E2-75/25", "E3-equal"):
        model = labels[short]
        failure_cases[short] = (
            predictions.loc[predictions["model"] == model]
            .sort_values("ae", ascending=False)
            .head(8)[["Date", "destination", "actual", "forecast", "ae", "horizon"]]
            .to_dict(orient="records")
        )

    horizon_validation = {}
    for short, model in labels.items():
        row = by_horizon.loc[by_horizon["model"] == model].sort_values("horizon")
        if row.empty:
            continue
        horizon_validation[short] = row[
            ["horizon", "mean_mase", "mean_mae", "mean_mape"]
        ].to_dict(orient="records")

    recovery_origins = ["2020M12", "2021M03"]
    backtest_recovery = {}
    for short, model in labels.items():
        if short.startswith("E"):
            continue
        # Validation short names differ from backtest labels; map by family params.
        if short in ("Lag1", "Lag12", "MA3", "MA6"):
            backtest_model = {
                "Lag1": "naive_lag1({})",
                "Lag12": "naive_lag12({})",
                "MA3": "recent_ma_3({})",
                "MA6": "recent_ma_6({})",
            }[short]
        else:
            family = {
                "Damped": "damped_trend",
                "Adaptive": "adaptive_seasonal",
                "Ridge": "global_ridge",
                "HGB": "global_hgb",
            }[short]
            backtest_model = _backtest_label(
                family, _parse_validation_name(model)
            )
        sub = frame[
            (frame["model"] == backtest_model)
            & (frame["origin"].isin(recovery_origins))
        ]
        if sub.empty:
            continue
        table = sub.groupby("horizon").agg(
            mean_mase=("mase", "mean"), mean_mae=("ae", "mean")
        ).reset_index()
        backtest_recovery[short] = table.to_dict(orient="records")

    facts = {
        "data": log["data"],
        "notebook_baseline_check": log["notebook_baseline_check"],
        "audit_isolation": log["audit_isolation"],
        "selected_candidates_by_backtest": log["selected_candidates_by_backtest"],
        "family_best_validation": family_best,
        "key_model_summary": key_models.to_dict(orient="records"),
        "destination_detail": destination_detail,
        "failure_cases": failure_cases,
        "horizon_validation": horizon_validation,
        "backtest_recovery_horizon": backtest_recovery,
        "backtest_ensemble_by_origin": ensemble_by_origin.to_dict(orient="records"),
        "backtest_ensemble_by_horizon": ensemble_by_horizon.to_dict(orient="records"),
        "backtest_ensemble_recovery_horizon": ensemble_recovery.to_dict(orient="records"),
        "backtest_origins": log["config"]["backtest"]["origins"],
        "backtest_runtime_seconds": log["backtest_runtime_seconds"],
        "total_runtime_seconds": log["total_runtime_seconds"],
        "seed": log["seed"],
    }
    (RESULTS / "report_facts.json").write_text(
        json.dumps(facts, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    pd.set_option("display.width", 220)
    print(key_models[
        ["short", "mean_MASE", "median_MASE", "MAE_macro", "mean_MAPE",
         "destinations_better_than_lag1", "destinations_worse_than_lag1",
         "worst_destination", "worst_destination_mase"]
    ].round(3).to_string(index=False))
    print("\nEnsemble backtest by origin:")
    print(ensemble_by_origin.pivot(index="origin", columns="model", values="mean_mase").round(3).to_string())
    print("\nEnsemble backtest by horizon (recovery origins):")
    print(
        ensemble_recovery.pivot(index="horizon", columns="model", values="mean_mase")
        .round(3).to_string()
    )
    print(f"\nWrote analysis tables to {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
