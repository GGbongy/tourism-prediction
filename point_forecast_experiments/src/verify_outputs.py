"""Acceptance checks for the point-forecasting experiment outputs."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .data_loader import load_data

EXP_DIR = Path(__file__).resolve().parents[1]
RESULTS = EXP_DIR / "results"
FIGURES = EXP_DIR / "figures"
REPORT = EXP_DIR / "reports" / "Point_Forecasting_Experiment_Report.md"


def main() -> int:
    failures: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    required = [
        "metrics_summary.csv",
        "metrics_by_destination.csv",
        "metrics_by_horizon.csv",
        "validation_predictions.csv",
        "backtest_metrics.csv",
        "runtime_comparison.csv",
        "experiment_log.json",
        "report_facts.json",
        "backtest_by_origin.csv",
        "backtest_by_horizon.csv",
        "backtest_ensemble_by_origin.csv",
        "backtest_ensemble_recovery_by_horizon.csv",
    ]
    for name in required:
        check((RESULTS / name).is_file(), f"missing result file: {name}")

    config = json.loads(
        (EXP_DIR / "configs" / "experiment_config.json").read_text(encoding="utf-8")
    )
    data = load_data(config, EXP_DIR.parent)
    destinations = data.destinations
    validation_months = data.validation_months

    summary = pd.read_csv(RESULTS / "metrics_summary.csv")
    by_destination = pd.read_csv(RESULTS / "metrics_by_destination.csv")
    predictions = pd.read_csv(RESULTS / "validation_predictions.csv")
    backtest = pd.read_csv(RESULTS / "backtest_metrics.csv")
    log = json.loads((RESULTS / "experiment_log.json").read_text(encoding="utf-8"))

    # Summary vs destination table.
    recomputed = by_destination.groupby("model", as_index=False).agg(
        mean_mase=("mase", "mean"), mean_mae=("mae", "mean")
    )
    merged = summary.merge(recomputed, on="model", how="left")
    check(len(merged) == len(summary), "summary/destination model sets differ")
    check(
        bool(np.allclose(merged["mean_MASE"], merged["mean_mase"])),
        "mean MASE differs between summary and destination tables",
    )
    check(
        bool(np.allclose(merged["MAE_macro"], merged["mean_mae"])),
        "macro MAE differs between summary and destination tables",
    )

    # Validation predictions.
    expected_keys = {
        (month, destination)
        for month in validation_months
        for destination in destinations
    }
    for model, frame in predictions.groupby("model"):
        check(len(frame) == 140, f"{model}: expected 140 validation rows, got {len(frame)}")
        keys = set(zip(frame["Date"], frame["destination"], strict=True))
        check(keys == expected_keys, f"{model}: validation keys are incomplete")
        check(
            bool(np.isfinite(frame[["actual", "forecast", "ae"]].to_numpy(float)).all()),
            f"{model}: non-finite validation values",
        )
        check(bool((frame["forecast"] >= 0).all()), f"{model}: negative forecast")

    actual_long = data.validation_actuals.melt(
        id_vars="Date", value_vars=list(destinations),
        var_name="destination", value_name="expected_actual",
    )
    paired = predictions.merge(actual_long, on=["Date", "destination"], how="left")
    check(
        bool(np.allclose(paired["actual"], paired["expected_actual"])),
        "validation actuals do not match the source split",
    )

    # Backtest leakage and search space.
    check(
        bool((backtest["Date"].map(lambda x: int(x[:4]) * 12 + int(x[-2:]))
              <= 2022 * 12 + 7).all()),
        "backtest contains targets after 2022M07",
    )
    check(
        set(backtest["origin"]) <= set(config["backtest"]["origins"]),
        "unexpected backtest origin",
    )
    check(
        bool(backtest["horizon"].between(1, 12).all()),
        "backtest horizon outside 1..12",
    )
    check(
        bool(np.isfinite(backtest[["forecast", "actual", "ae", "mase"]].to_numpy(float)).all()),
        "backtest contains non-finite values",
    )
    check(bool((backtest["forecast"] >= 0).all()), "backtest contains negative forecasts")

    # Isolation and baseline reproduction.
    check(log.get("audit_used") is False, "experiment log says audit was used")
    check(
        log["data"]["history_date_range"][1] == "2022M07",
        "training history extends past 2022M07",
    )
    check(
        all(item["matches"] for item in log["notebook_baseline_check"].values()),
        "notebook baseline reproduction failed",
    )

    # No formal submission files.
    check(not (EXP_DIR / "Forecast.csv").exists(), "unexpected Forecast.csv created")
    check(not (EXP_DIR / "Intervals.csv").exists(), "unexpected Intervals.csv created")

    # Report and figure links.
    check(REPORT.is_file(), "report is missing")
    report_text = REPORT.read_text(encoding="utf-8")
    for link in re.findall(r"\]\((\.\./figures/[^)]+)\)", report_text):
        check((REPORT.parent / link).resolve().is_file(), f"broken figure link: {link}")
    for token in (
        "0.8051", "0.8383", "0.8163", "0.8500", "0.8669", "1.4682",
        "31,297.75", "43,752.56", "0.471", "0.443", "0.393", "0.383", "0.593",
    ):
        check(token in report_text, f"report is missing expected value: {token}")

    if failures:
        print("FAIL")
        for message in failures:
            print(f"ERROR: {message}")
        return 1
    print(
        f"PASS: {summary['model'].nunique()} models, "
        f"{len(predictions)} validation rows, {len(backtest)} backtest rows"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
