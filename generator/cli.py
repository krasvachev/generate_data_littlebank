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


def load_config(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def resolve_range(column, config_ranges):
    """Return the (min, max) range for a numeric column.

    Uses the config override if present, else the schema default.
    """
    override = config_ranges.get(column.name)
    if override is not None:
        return float(override["min"]), float(override["max"])
    return column.default_range


def _resolve_distribution(column, config, *, block, global_key, params_key, default):
    """Return the (distribution, params) for a column.

    Precedence: per-column entry in ``block`` > global ``global_key`` >
    ``default``. Per-column 'params' override the global ``params_key``.
    """
    override = config.get(block, {}).get(column.name, {})
    distribution = override.get("distribution", config.get(global_key, default))
    params = {**config.get(params_key, {}), **override.get("params", {})}
    return distribution, params


def resolve_distribution(column, config):
    """Return the (distribution, params) for a numeric column.

    Precedence: per-column 'distribution' in ranges > global 'distribution' >
    the built-in default. Per-column 'params' override global 'params'.
    """
    return _resolve_distribution(
        column, config, block="ranges", global_key="distribution",
        params_key="params", default=gen.DEFAULT_DISTRIBUTION,
    )


def resolve_categorical_distribution(column, config):
    """Return the (distribution, params) for a categorical column.

    Precedence: per-column entry in the 'categorical' block >
    global 'categorical_distribution' > the default ('uniform', preserving the
    original behaviour). Per-column 'params' override global 'categorical_params'.
    """
    return _resolve_distribution(
        column, config, block="categorical", global_key="categorical_distribution",
        params_key="categorical_params", default=gen.UNIFORM,
    )


def _check_distribution(dist, label):
    """Raise ValueError if dist is set and not a known distribution."""
    if dist is not None and dist not in gen.DISTRIBUTIONS:
        raise ValueError(
            f"unknown {label}: {dist!r}. Choose from {sorted(gen.DISTRIBUTIONS)}"
        )


def validate_config(config):
    rows = config.get("rows")
    if not isinstance(rows, int) or rows <= 0:
        raise ValueError("config 'rows' must be a positive integer")

    ratio = config.get("outcome_true_ratio")
    if not isinstance(ratio, (int, float)) or not 0 <= ratio <= 1:
        raise ValueError("config 'outcome_true_ratio' must be between 0 and 1")

    _check_distribution(
        config.get("distribution", gen.DEFAULT_DISTRIBUTION), "'distribution'"
    )
    _check_distribution(
        config.get("categorical_distribution", gen.UNIFORM),
        "'categorical_distribution'",
    )

    for name, entry in config.get("categorical", {}).items():
        col = COLUMNS_BY_NAME.get(name)
        if col is None or col.kind != KIND_CATEGORICAL:
            raise ValueError(
                f"config 'categorical' has unknown/non-categorical column: {name!r}"
            )
        _check_distribution(
            entry.get("distribution"), f"categorical distribution for {name!r}"
        )

    config_ranges = config.get("ranges", {})
    for name in config_ranges:
        col = COLUMNS_BY_NAME.get(name)
        if col is None or not col.is_numeric:
            raise ValueError(
                f"config 'ranges' has unknown/non-numeric column: {name!r}"
            )
        rng = config_ranges[name]
        if "min" not in rng or "max" not in rng:
            raise ValueError(f"range for {name!r} needs both 'min' and 'max'")
        if float(rng["min"]) > float(rng["max"]):
            raise ValueError(f"range for {name!r} has min > max")
        _check_distribution(rng.get("distribution"), f"distribution for {name!r}")


def build_dataframe(config):
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


def output_path(config):
    base = config.get("output_base", DEFAULT_OUTPUT_BASE)
    date_str = dt.date.today().isoformat()
    return f"{base}_{date_str}.csv"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate synthetic littlebank CSV data."
    )
    parser.add_argument(
        "--config", required=True, help="Path to the JSON config file."
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    validate_config(config)

    df = build_dataframe(config)

    path = output_path(config)
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
