"""Student-safe structural validation for the tourism demand forecasting project outputs.

This module contains no hidden actuals, marking bands, or evaluator logic. It
is the canonical source for the validator that may later be promoted into the
public Lab repository after release approval.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


DESTINATIONS = (
    "Canada",
    "Chile",
    "Mexico",
    "Taiwan China",
    "Hong Kong China",
    "Japan",
    "South Korea",
    "Macao China",
    "Maldives",
    "Cambodia",
    "Indonesia",
    "Singapore",
    "New Zealand",
    "USA",
    "Thailand",
    "Turkey",
    "Australia",
    "Hawaii",
    "Austria",
    "Czech",
)
FORECAST_DATES = tuple(
    [f"2023M{month:02d}" for month in range(8, 13)]
    + [f"2024M{month:02d}" for month in range(1, 8)]
)
FORECAST_COLUMNS = ("Date", *DESTINATIONS)
INTERVAL_COLUMNS = (
    "Date",
    "destination",
    "point",
    "lower80",
    "upper80",
    "lower95",
    "upper95",
    "model_label",
)
EXPECTED_CASES = len(DESTINATIONS) * len(FORECAST_DATES)


@dataclass
class ValidationResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    valid_point_cases: set[tuple[str, str]] = field(default_factory=set)
    valid_interval_cases: set[tuple[str, str]] = field(default_factory=set)
    points: dict[tuple[str, str], float] = field(default_factory=dict)
    intervals: dict[tuple[str, str], dict[str, float | str]] = field(
        default_factory=dict
    )

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "expected_cases": EXPECTED_CASES,
            "valid_point_cases": len(self.valid_point_cases),
            "point_completeness": len(self.valid_point_cases) / EXPECTED_CASES,
            "valid_interval_cases": len(self.valid_interval_cases),
            "interval_completeness": len(self.valid_interval_cases)
            / EXPECTED_CASES,
            "errors": self.errors,
            "warnings": self.warnings,
        }


def _finite_nonnegative(raw: str | None, label: str, errors: list[str]) -> float | None:
    if raw is None or raw.strip() == "":
        errors.append(f"{label}: value is missing")
        return None
    try:
        value = float(raw)
    except ValueError:
        errors.append(f"{label}: {raw!r} is not numeric")
        return None
    if not math.isfinite(value):
        errors.append(f"{label}: value must be finite")
        return None
    if value < 0:
        errors.append(f"{label}: value must be non-negative")
        return None
    return value


def _header_error(actual: Iterable[str] | None, expected: tuple[str, ...], name: str) -> str | None:
    actual_tuple = tuple(actual or ())
    if actual_tuple == expected:
        return None
    missing = [column for column in expected if column not in actual_tuple]
    extra = [column for column in actual_tuple if column not in expected]
    return (
        f"{name}: columns must exactly match the published order; "
        f"missing={missing}, extra={extra}, actual={list(actual_tuple)}"
    )


def validate_forecast(path: str | Path, result: ValidationResult | None = None) -> ValidationResult:
    result = result or ValidationResult()
    forecast_path = Path(path)
    try:
        handle = forecast_path.open("r", encoding="utf-8-sig", newline="")
    except OSError as exc:
        result.errors.append(f"Forecast.csv: cannot read {forecast_path}: {exc}")
        return result

    with handle:
        reader = csv.DictReader(handle)
        header_problem = _header_error(reader.fieldnames, FORECAST_COLUMNS, "Forecast.csv")
        if header_problem:
            result.errors.append(header_problem)
        rows = list(reader)

    by_date: dict[str, list[dict[str, str]]] = {}
    for row_number, row in enumerate(rows, start=2):
        date = (row.get("Date") or "").strip()
        if date not in FORECAST_DATES:
            result.errors.append(
                f"Forecast.csv row {row_number}: unexpected Date {date!r}"
            )
            continue
        by_date.setdefault(date, []).append(row)

    for date in FORECAST_DATES:
        date_rows = by_date.get(date, [])
        if not date_rows:
            result.errors.append(f"Forecast.csv: missing row for {date}")
            continue
        if len(date_rows) > 1:
            result.errors.append(f"Forecast.csv: duplicate rows for {date}")
            continue
        row = date_rows[0]
        for destination in DESTINATIONS:
            key = (date, destination)
            local_errors: list[str] = []
            value = _finite_nonnegative(
                row.get(destination),
                f"Forecast.csv {date}/{destination}",
                local_errors,
            )
            result.errors.extend(local_errors)
            if value is not None:
                result.points[key] = value
                result.valid_point_cases.add(key)

    if len(rows) != len(FORECAST_DATES):
        result.errors.append(
            f"Forecast.csv: expected {len(FORECAST_DATES)} data rows, found {len(rows)}"
        )
    return result


def validate_intervals(path: str | Path, result: ValidationResult) -> ValidationResult:
    interval_path = Path(path)
    try:
        handle = interval_path.open("r", encoding="utf-8-sig", newline="")
    except OSError as exc:
        result.errors.append(f"Intervals.csv: cannot read {interval_path}: {exc}")
        return result

    with handle:
        reader = csv.DictReader(handle)
        header_problem = _header_error(reader.fieldnames, INTERVAL_COLUMNS, "Intervals.csv")
        if header_problem:
            result.errors.append(header_problem)
        rows = list(reader)

    expected_keys = {(date, destination) for date in FORECAST_DATES for destination in DESTINATIONS}
    seen: dict[tuple[str, str], list[tuple[int, dict[str, str]]]] = {}
    for row_number, row in enumerate(rows, start=2):
        date = (row.get("Date") or "").strip()
        destination = (row.get("destination") or "").strip()
        if date not in FORECAST_DATES or destination not in DESTINATIONS:
            result.errors.append(
                f"Intervals.csv row {row_number}: unexpected key "
                f"Date={date!r}, destination={destination!r}"
            )
            continue
        seen.setdefault((date, destination), []).append((row_number, row))

    for key in sorted(expected_keys):
        matches = seen.get(key, [])
        if not matches:
            result.errors.append(f"Intervals.csv: missing row for {key[0]}/{key[1]}")
            continue
        if len(matches) > 1:
            result.errors.append(f"Intervals.csv: duplicate rows for {key[0]}/{key[1]}")
            continue
        row_number, row = matches[0]
        local_errors: list[str] = []
        values: dict[str, float] = {}
        for column in ("point", "lower80", "upper80", "lower95", "upper95"):
            parsed = _finite_nonnegative(
                row.get(column),
                f"Intervals.csv row {row_number} {column}",
                local_errors,
            )
            if parsed is not None:
                values[column] = parsed
        model_label = (row.get("model_label") or "").strip()
        if not model_label:
            local_errors.append(
                f"Intervals.csv row {row_number}: model_label must not be blank"
            )
        elif "," in model_label or "\n" in model_label or "\r" in model_label:
            local_errors.append(
                f"Intervals.csv row {row_number}: model_label must not contain a comma or line break"
            )
        if len(values) == 5:
            ordered = (
                values["lower95"],
                values["lower80"],
                values["point"],
                values["upper80"],
                values["upper95"],
            )
            if ordered != tuple(sorted(ordered)):
                local_errors.append(
                    f"Intervals.csv row {row_number}: bounds must satisfy "
                    "lower95 <= lower80 <= point <= upper80 <= upper95"
                )
            forecast_point = result.points.get(key)
            if forecast_point is None:
                local_errors.append(
                    f"Intervals.csv row {row_number}: matching Forecast.csv point is invalid or missing"
                )
            elif values["point"] != forecast_point:
                local_errors.append(
                    f"Intervals.csv row {row_number}: point {values['point']} does not "
                    f"exactly match Forecast.csv value {forecast_point}"
                )
        result.errors.extend(local_errors)
        if not local_errors:
            result.intervals[key] = {**values, "model_label": model_label}
            result.valid_interval_cases.add(key)

    if len(rows) != EXPECTED_CASES:
        result.errors.append(
            f"Intervals.csv: expected {EXPECTED_CASES} data rows, found {len(rows)}"
        )
    return result


def validate_submission(
    forecast_path: str | Path, interval_path: str | Path
) -> ValidationResult:
    result = validate_forecast(forecast_path)
    return validate_intervals(interval_path, result)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate the tourism demand forecasting project Forecast.csv and Intervals.csv."
    )
    parser.add_argument("forecast", type=Path)
    parser.add_argument("intervals", type=Path)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)
    result = validate_submission(args.forecast, args.intervals)
    if args.as_json:
        print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    else:
        summary = result.as_dict()
        print(
            f"Point cases: {summary['valid_point_cases']}/{EXPECTED_CASES}; "
            f"interval cases: {summary['valid_interval_cases']}/{EXPECTED_CASES}"
        )
        for message in result.errors:
            print(f"ERROR: {message}")
        for message in result.warnings:
            print(f"WARNING: {message}")
        print("PASS" if result.ok else "FAIL")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
