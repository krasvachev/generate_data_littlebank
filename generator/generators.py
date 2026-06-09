"""Per-column value generators.

Numeric columns use a normal distribution centred on the midpoint of their
range with std = (max - min) / 6, clipped to the range bounds. Categorical
columns are sampled uniformly. The unique-ID column is sequential, and the
boolean target is produced from an exact class ratio and shuffled.
"""

import numpy as np

from .schema import FLOAT_DECIMALS


def sequential_ids(rows, start=1):
    """Unique sequential integer IDs: start, start+1, ..."""
    return np.arange(start, start + rows, dtype=np.int64)


def _normal_clipped(rows, low, high):
    """Normal sample centred on the midpoint, clipped to [low, high]."""
    mean = (low + high) / 2.0
    std = (high - low) / 6.0
    if std == 0:
        return np.full(rows, mean, dtype=float)
    values = np.random.normal(loc=mean, scale=std, size=rows)
    return np.clip(values, low, high)


def integers(rows, low, high):
    """Integers in [low, high] from a clipped normal distribution."""
    values = _normal_clipped(rows, low, high)
    return np.rint(values).astype(np.int64)


def floats(rows, low, high, decimals=FLOAT_DECIMALS):
    """Floats in [low, high] from a clipped normal distribution, rounded."""
    values = _normal_clipped(rows, low, high)
    return np.round(values, decimals)


def categorical(rows, categories):
    """Uniformly sampled categorical values."""
    return np.random.choice(np.asarray(categories, dtype=object), size=rows)


def boolean_by_ratio(rows, true_ratio):
    """Boolean array with round(rows * true_ratio) Trues, shuffled."""
    n_true = int(round(rows * true_ratio))
    values = np.zeros(rows, dtype=bool)
    values[:n_true] = True
    np.random.shuffle(values)
    return values


def enforce_greater_than(high_vals, low_vals, high_low, high_high):
    """Ensure high_vals > low_vals element-wise within [high_low, high_high].

    Rows that violate the rule are resampled uniformly from
    (low + 1) .. high_high, which always has room because the high column's
    upper bound exceeds the low column's upper bound.
    """
    high_vals = high_vals.copy()
    violating = high_vals <= low_vals
    if violating.any():
        lower_bounds = np.maximum(low_vals[violating] + 1, high_low)
        high_vals[violating] = np.array([
            np.random.randint(lb, high_high + 1)
            for lb in lower_bounds
        ], dtype=high_vals.dtype)
    return high_vals
