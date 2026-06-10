"""Command-line entry point: read a JSON config, build the dataset, write CSV.

Usage:
    python -m generator.cli --config config.example.json
"""

import argparse
import datetime as dt
import json
import os

import pandas as pd

from . import generators as gen
from .schema import (
    COLUMNS,
    COLUMN_ORDER,
    Column,
    GREATER_THAN_RULES,
    KIND_BOOL,
    KIND_CATEGORICAL,
    KIND_FLOAT,
    KIND_ID,
    KIND_INT,
    TARGET_COLUMN,
    COLUMNS_BY_NAME,
)

DEFAULT_OUTPUT_BASE = "output/littlebank"


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_range(column: Column, config_ranges: dict) -> tuple[float, float]:
    """Return the (min, max) range for a numeric column.

    Uses the config override if present, else the schema default.
    """
    override = config_ranges.get(column.name)
    if override is not None:
        return float(override["min"]), float(override["max"])
    return column.default_range


def _resolve_distribution(column: Column, config: dict, *, block: str,
                          global_key: str, params_key: str,
                          default: str) -> tuple[str, dict]:
    """Return the (distribution, params) for a column.

    Precedence: per-column entry in ``block`` > global ``global_key`` >
    ``default``. Per-column 'params' override the global ``params_key``.
    """
    override = config.get(block, {}).get(column.name, {})
    distribution = override.get("distribution", config.get(global_key, default))
    params = {**config.get(params_key, {}), **override.get("params", {})}
    return distribution, params


def resolve_distribution(column: Column, config: dict) -> tuple[str, dict]:
    """Return the (distribution, params) for a numeric column.

    Precedence: per-column 'distribution' in ranges > global 'distribution' >
    the built-in default. Per-column 'params' override global 'params'.
    """
    return _resolve_distribution(
        column, config, block="ranges", global_key="distribution",
        params_key="params", default=gen.DEFAULT_DISTRIBUTION,
    )


def resolve_categorical_distribution(column: Column, config: dict) -> tuple[str, dict]:
    """Return the (distribution, params) for a categorical column.

    Precedence: per-column entry in the 'categorical' block >
    global 'categorical_distribution' > the default ('uniform', preserving the
    original behaviour). Per-column 'params' override global 'categorical_params'.
    """
    return _resolve_distribution(
        column, config, block="categorical", global_key="categorical_distribution",
        params_key="categorical_params", default=gen.UNIFORM,
    )


def _check_distribution(distribution, label: str) -> None:
    """Raise ValueError if distribution is set and not a known distribution."""
    if distribution is not None and distribution not in gen.DISTRIBUTIONS:
        raise ValueError(
            f"unknown {label}: {distribution!r}. "
            f"Choose from {sorted(gen.DISTRIBUTIONS)}"
        )


def validate_config(config: dict) -> None:
    rows = config.get("rows")
    if not isinstance(rows, int) or rows <= 0:
        raise ValueError("config 'rows' must be a positive integer")

    ratio = config.get("outcome_true_ratio")
    if not isinstance(ratio, (int, float)) or not (0 <= ratio <= 1):
        raise ValueError("config 'outcome_true_ratio' must be between 0 and 1")

    _check_distribution(
        config.get("distribution", gen.DEFAULT_DISTRIBUTION), "'distribution'"
    )
    _check_distribution(
        config.get("categorical_distribution", gen.UNIFORM),
        "'categorical_distribution'",
    )

    for name, entry in config.get("categorical", {}).items():
        column = COLUMNS_BY_NAME.get(name)
        if column is None or column.kind != KIND_CATEGORICAL:
            raise ValueError(
                f"config 'categorical' has unknown/non-categorical column: {name!r}"
            )
        _check_distribution(
            entry.get("distribution"), f"categorical distribution for {name!r}"
        )

    config_ranges = config.get("ranges", {})
    for name in config_ranges:
        column = COLUMNS_BY_NAME.get(name)
        if column is None or not column.is_numeric:
            raise ValueError(
                f"config 'ranges' has unknown/non-numeric column: {name!r}"
            )
        range_cfg = config_ranges[name]
        if "min" not in range_cfg or "max" not in range_cfg:
            raise ValueError(f"range for {name!r} needs both 'min' and 'max'")
        if float(range_cfg["min"]) > float(range_cfg["max"]):
            raise ValueError(f"range for {name!r} has min > max")
        _check_distribution(
            range_cfg.get("distribution"), f"distribution for {name!r}"
        )


def build_dataframe(config: dict) -> pd.DataFrame:
    rows = config["rows"]
    ratio = config["outcome_true_ratio"]
    config_ranges = config.get("ranges", {})

    data = {}
    for column in COLUMNS:
        if column.kind == KIND_ID:
            data[column.name] = gen.sequential_ids(rows)
        elif column.kind == KIND_CATEGORICAL:
            distribution, params = resolve_categorical_distribution(column, config)
            data[column.name] = gen.categorical(
                rows, column.categories, distribution, params
            )
        elif column.kind in (KIND_INT, KIND_FLOAT):
            low, high = resolve_range(column, config_ranges)
            distribution, params = resolve_distribution(column, config)
            if column.kind == KIND_INT:
                low, high = int(low), int(high)
            data[column.name] = gen.numeric(
                rows, low, high, column.kind, distribution, params
            )
        elif column.kind == KIND_BOOL:
            data[column.name] = gen.boolean_by_ratio(rows, ratio)

    # Enforce cross-column rules (e.g. high_temp > low_temp).
    for high_name, low_name in GREATER_THAN_RULES:
        high_col = COLUMNS_BY_NAME[high_name]
        low, high = resolve_range(high_col, config_ranges)
        data[high_name] = gen.enforce_greater_than(
            data[high_name], data[low_name], int(low), int(high)
        )

    return pd.DataFrame(data, columns=COLUMN_ORDER)


def build_output_path(config: dict) -> str:
    base = config.get("output_base", DEFAULT_OUTPUT_BASE)
    date_str = dt.date.today().isoformat()
    return f"{base}_{date_str}.csv"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate synthetic LittleBank Case Study CSV data."
    )
    parser.add_argument(
        "--config", required=True, help="Path to the JSON config file."
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    validate_config(config)

    df = build_dataframe(config)

    path = build_output_path(config)
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")

    true_count = int(df[TARGET_COLUMN].sum())
    print(
        f"Wrote {len(df)} rows to {path} "
        f"({true_count} True / {len(df) - true_count} False on {TARGET_COLUMN})"
    )
    return path


if __name__ == "__main__":
    main()
