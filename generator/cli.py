"""Command-line entry point: read a JSON config, build the dataset, write CSV.

Usage:
    python -m generator.cli --config config.example.json
"""

import argparse
import datetime as dt
import json
import os

import numpy as np
import pandas as pd

from . import generators as gen
from .schema import (
    COLUMNS,
    COLUMN_ORDER,
    GREATER_THAN_RULES,
    ID_COLUMN,
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


def resolve_distribution(column, config):
    """Return the (distribution, params) for a numeric column.

    Precedence: per-column 'distribution' in ranges > global 'distribution' >
    the built-in default. Per-column 'params' override global 'params'.
    """
    config_ranges = config.get("ranges", {})
    override = config_ranges.get(column.name, {})

    distribution = override.get(
        "distribution",
        config.get("distribution", gen.DEFAULT_DISTRIBUTION),
    )
    params = dict(config.get("params", {}))
    params.update(override.get("params", {}))
    return distribution, params


def resolve_outcome_weights(config, data, rows):
    """Return per-row selection weights for the outcome target, or None.

    Returns None when the config has no 'outcome_model', meaning the outcome
    should be generated with the default uniform behaviour. Otherwise returns a
    float array of length `rows` where each entry is the product of the
    per-category weights of that row's driver columns (categories not listed in
    the model default to weight 1.0).
    """
    model = config.get("outcome_model")
    if not model:
        return None

    weights = np.ones(rows, dtype=float)
    for col_name, group_weights in model.get("weights", {}).items():
        values = np.asarray(data[col_name], dtype=object)
        mapped = np.array(
            [float(group_weights.get(v, 1.0)) for v in values], dtype=float
        )
        weights = weights * mapped
    return weights


def validate_config(config):
    rows = config.get("rows")
    if not isinstance(rows, int) or rows <= 0:
        raise ValueError("config 'rows' must be a positive integer")

    ratio = config.get("outcome_true_ratio")
    if not isinstance(ratio, (int, float)) or not 0 <= ratio <= 1:
        raise ValueError("config 'outcome_true_ratio' must be between 0 and 1")

    global_dist = config.get("distribution", gen.DEFAULT_DISTRIBUTION)
    if global_dist not in gen.DISTRIBUTIONS:
        raise ValueError(
            f"unknown 'distribution': {global_dist!r}. "
            f"Choose from {sorted(gen.DISTRIBUTIONS)}"
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
        dist = rng.get("distribution")
        if dist is not None and dist not in gen.DISTRIBUTIONS:
            raise ValueError(
                f"unknown distribution for {name!r}: {dist!r}. "
                f"Choose from {sorted(gen.DISTRIBUTIONS)}"
            )

    model = config.get("outcome_model")
    if model is not None:
        if not isinstance(model, dict):
            raise ValueError("config 'outcome_model' must be an object")
        weights = model.get("weights", {})
        if not isinstance(weights, dict):
            raise ValueError("config 'outcome_model.weights' must be an object")
        for name, group_weights in weights.items():
            col = COLUMNS_BY_NAME.get(name)
            if col is None or col.kind != KIND_CATEGORICAL:
                raise ValueError(
                    "config 'outcome_model.weights' has unknown/non-categorical "
                    f"column: {name!r}"
                )
            if not isinstance(group_weights, dict):
                raise ValueError(
                    f"outcome_model weights for {name!r} must be an object "
                    "mapping category -> number"
                )
            for category, weight in group_weights.items():
                if category not in col.categories:
                    raise ValueError(
                        f"outcome_model weights for {name!r} has unknown "
                        f"category: {category!r}"
                    )
                if not isinstance(weight, (int, float)) or isinstance(weight, bool):
                    raise ValueError(
                        f"outcome_model weight for {name!r}/{category!r} must "
                        "be a number"
                    )
                if weight < 0:
                    raise ValueError(
                        f"outcome_model weight for {name!r}/{category!r} must "
                        "not be negative"
                    )


def build_dataframe(config):
    rows = config["rows"]
    ratio = config["outcome_true_ratio"]
    config_ranges = config.get("ranges", {})

    data = {}
    for column in COLUMNS:
        if column.kind == KIND_ID:
            data[column.name] = gen.sequential_ids(rows)
        elif column.kind == KIND_CATEGORICAL:
            data[column.name] = gen.categorical(rows, column.categories)
        elif column.kind in (KIND_INT, KIND_FLOAT):
            low, high = resolve_range(column, config_ranges)
            distribution, params = resolve_distribution(column, config)
            if column.kind == KIND_INT:
                low, high = int(low), int(high)
            data[column.name] = gen.numeric(
                rows, low, high, column.kind, distribution, params
            )
        elif column.kind == KIND_BOOL:
            weights = resolve_outcome_weights(config, data, rows)
            if weights is None:
                data[column.name] = gen.boolean_by_ratio(rows, ratio)
            else:
                data[column.name] = gen.boolean_dependent(rows, ratio, weights)

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
