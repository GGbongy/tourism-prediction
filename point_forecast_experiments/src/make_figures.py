"""Generate figures for the point-forecasting experiment report."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .analyze_results import (  # noqa: E402
    RESULTS,
    _backtest_label,
    _parse_validation_name,
    _short_labels,
    load_backtest,
)

FIGURES = RESULTS.parent / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


def _validation_tables():
    summary = pd.read_csv(RESULTS / "metrics_summary.csv")
    by_horizon = pd.read_csv(RESULTS / "metrics_by_horizon.csv")
    by_destination = pd.read_csv(RESULTS / "metrics_by_destination.csv")
    predictions = pd.read_csv(RESULTS / "validation_predictions.csv")
    return summary, by_horizon, by_destination, predictions


def figure_overall(summary: pd.DataFrame, labels: dict[str, str]) -> None:
    rows = []
    for short, model in labels.items():
        match = summary.loc[summary["model"] == model]
        if not match.empty:
            rows.append((short, float(match.iloc[0]["mean_MASE"])))
    frame = pd.DataFrame(rows, columns=["model", "mean_MASE"]).sort_values(
        "mean_MASE", ascending=True
    )
    plt.figure(figsize=(9, 5))
    colors = ["#2f6f4e" if name == "MA3" else "#7a8fa6" for name in frame["model"]]
    colors = ["#b45309" if name == "E3-equal" else color for name, color in zip(frame["model"], colors)]
    bars = plt.barh(frame["model"], frame["mean_MASE"], color=colors)
    plt.axvline(0.866913, color="#555555", linestyle="--", linewidth=1, label="Lag-1 baseline")
    plt.axvline(1.413921, color="#999999", linestyle=":", linewidth=1, label="Lag-12 baseline")
    for bar, value in zip(bars, frame["mean_MASE"]):
        plt.text(value + 0.012, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=8)
    plt.xlabel("Validation mean MASE (lower is better)")
    plt.title("Validation mean MASE by model")
    plt.legend(loc="lower right", fontsize=8)
    plt.xlim(0, max(1.6, frame["mean_MASE"].max() * 1.12))
    plt.tight_layout()
    plt.savefig(FIGURES / "fig_overall_mean_mase.png", dpi=160)
    plt.close()


def figure_heatmap(by_destination: pd.DataFrame, labels: dict[str, str]) -> None:
    selected = ["MA3", "E4-equal", "E3-equal", "E2-75/25", "Ridge", "Damped", "HGB", "Lag1", "Lag12"]
    columns = []
    for short in selected:
        model = labels[short]
        frame = by_destination.loc[by_destination["model"] == model].set_index("destination")
        columns.append(frame["mase"].rename(short))
    matrix = pd.concat(columns, axis=1)
    order = matrix["MA3"].sort_values(ascending=False).index
    matrix = matrix.loc[order]
    plt.figure(figsize=(10, 8))
    image = plt.imshow(matrix.to_numpy(), aspect="auto", cmap="YlOrRd", vmin=0, vmax=4.5)
    plt.colorbar(image, label="MASE (clipped at 4.5)")
    plt.xticks(range(len(matrix.columns)), matrix.columns, rotation=35, ha="right")
    plt.yticks(range(len(matrix.index)), matrix.index)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix.iloc[row, col]
            plt.text(col, row, f"{value:.1f}", ha="center", va="center", fontsize=6,
                     color="black" if value < 3 else "white")
    plt.title("Validation MASE by destination")
    plt.tight_layout()
    plt.savefig(FIGURES / "fig_by_destination_heatmap.png", dpi=160)
    plt.close()


def figure_validation_horizon(by_horizon: pd.DataFrame, labels: dict[str, str]) -> None:
    selected = ["MA3", "E4-equal", "E3-equal", "E2-75/25", "Ridge", "Lag1", "Lag12"]
    plt.figure(figsize=(9, 5))
    for short in selected:
        model = labels[short]
        frame = by_horizon.loc[by_horizon["model"] == model].sort_values("horizon")
        if frame.empty:
            continue
        style = "--" if short == "Lag1" else ":"
        width = 2.6 if short in ("MA3", "E3-equal") else 1.6
        plt.plot(frame["horizon"], frame["mean_mase"], marker="o", linewidth=width,
                 linestyle=style if short in ("Lag1", "Lag12") else "-", label=short)
    plt.xlabel("Forecast horizon (months)")
    plt.ylabel("Validation mean MASE")
    plt.title("Validation MASE by forecast horizon")
    plt.xticks(range(1, 8))
    plt.grid(alpha=0.25)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "fig_validation_mase_by_horizon.png", dpi=160)
    plt.close()


def figure_backtest_recovery(labels: dict[str, str]) -> None:
    frame = load_backtest()
    recovery = ["2020M12", "2021M03"]
    selected = ["MA3", "Damped", "Ridge", "HGB", "Lag1"]
    plt.figure(figsize=(9, 5))
    for short in selected:
        model = labels[short]
        if short in ("MA3",):
            backtest_model = "recent_ma_3({})"
        elif short == "Lag1":
            backtest_model = "naive_lag1({})"
        else:
            family = {"Damped": "damped_trend", "Ridge": "global_ridge", "HGB": "global_hgb"}[short]
            backtest_model = _backtest_label(family, _parse_validation_name(model))
        sub = frame[
            (frame["model"] == backtest_model) & (frame["origin"].isin(recovery))
        ]
        if sub.empty:
            continue
        table = sub.groupby("horizon")["mase"].mean()
        plt.plot(table.index, table.values, marker="o", linewidth=2 if short == "MA3" else 1.5, label=short)
    ensemble = pd.read_csv(RESULTS / "backtest_ensemble_recovery_by_horizon.csv")
    for name in ["E2-75/25", "E3-equal", "E4-equal"]:
        sub = ensemble.loc[ensemble["model"] == name].sort_values("horizon")
        plt.plot(sub["horizon"], sub["mean_mase"], marker="s", linewidth=2 if name == "E3-equal" else 1.4,
                 linestyle="--", label=name)
    plt.xlabel("Forecast horizon (months)")
    plt.ylabel("Backtest mean MASE (recovery origins 2020M12, 2021M03)")
    plt.title("Long-horizon backtest stability during the recovery regime")
    plt.xticks(range(1, 13))
    plt.grid(alpha=0.25)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "fig_backtest_recovery_horizon.png", dpi=160)
    plt.close()


def figure_macao(predictions: pd.DataFrame, labels: dict[str, str]) -> None:
    selected = ["MA3", "E2-75/25", "E3-equal", "Ridge", "Lag1"]
    sub = predictions[predictions["destination"] == "Macao China"]
    actual = sub.loc[sub["model"] == labels["MA3"]].sort_values("Date")
    plt.figure(figsize=(9, 5))
    plt.plot(actual["Date"], actual["actual"], color="black", marker="o", linewidth=2.5, label="Actual")
    for short in selected:
        model = labels[short]
        frame = sub.loc[sub["model"] == model].sort_values("Date")
        if frame.empty:
            continue
        plt.plot(frame["Date"], frame["forecast"], marker="o", linewidth=1.8, label=short)
    plt.yscale("log")
    plt.ylabel("Monthly demand (log scale)")
    plt.title("Macao China: the dominant failure case")
    plt.xticks(rotation=30)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "fig_macao_failure.png", dpi=160)
    plt.close()


def main() -> int:
    summary, by_horizon, by_destination, predictions = _validation_tables()
    labels = _short_labels()
    figure_overall(summary, labels)
    figure_heatmap(by_destination, labels)
    figure_validation_horizon(by_horizon, labels)
    figure_backtest_recovery(labels)
    figure_macao(predictions, labels)
    print(f"Saved figures to {FIGURES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
