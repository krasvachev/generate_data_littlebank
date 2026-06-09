"""Per-column value generators.

Numeric columns are drawn from one of six selectable distributions, then
clipped to the column's range and rounded according to its type (ints to whole
numbers, floats to FLOAT_DECIMALS places). Categorical columns are sampled
uniformly, the unique-ID column is sequential, and the boolean target is
produced from an exact class ratio and shuffled.

Supported distributions (see DISTRIBUTIONS):
    normal       - Gaussian centred on the range midpoint (std = range/6).
    uniform      - uniform across [min, max].
    exponential  - right-skewed, anchored at min (scale = range/3).
    poisson      - discrete counts, anchored at min (lam = range/2).
    binomial     - discrete, anchored at min (n = round(range), p = 0.5).
    random       - alias for uniform within range (the "absolute random" mode).

Each distribution accepts optional per-column parameters (see _sample).
"""

import numpy as np

from .schema import KIND_INT, FLOAT_DECIMALS

NORMAL = "normal"
UNIFORM = "uniform"
EXPONENTIAL = "exponential"
POISSON = "poisson"
BINOMIAL = "binomial"
RANDOM = "random"  # absolute random == uniform within range

DISTRIBUTIONS = {NORMAL, UNIFORM, EXPONENTIAL, POISSON, BINOMIAL, RANDOM}

DEFAULT_DISTRIBUTION = NORMAL


def sequential_ids(rows, start=1):
    """Unique sequential integer IDs: start, start+1, ..."""
    return np.arange(start, start + rows, dtype=np.int64)


def _sample(distribution, rows, low, high, params):
    """Draw raw (unclipped, unrounded) values for the given distribution."""
    span = high - low
    if distribution == NORMAL:
        mean = params.get("mean", (low + high) / 2.0)
        std = params.get("std", span / 6.0)
        if std == 0:
            return np.full(rows, mean, dtype=float)
        return np.random.normal(loc=mean, scale=std, size=rows)
    if distribution in (UNIFORM, RANDOM):
        return np.random.uniform(low, high, size=rows)
    if distribution == EXPONENTIAL:
        scale = params.get("scale", span / 3.0 if span else 1.0)
        return low + np.random.exponential(scale=scale, size=rows)
    if distribution == POISSON:
        lam = params.get("lam", span / 2.0)
        return low + np.random.poisson(lam=lam, size=rows)
    if distribution == BINOMIAL:
        n = params.get("n", max(int(round(span)), 1))
        p = params.get("p", 0.5)
        return low + np.random.binomial(n=n, p=p, size=rows)
    raise ValueError(f"unknown distribution: {distribution!r}")


def numeric(rows, low, high, kind, distribution=DEFAULT_DISTRIBUTION, params=None):
    """Generate a numeric column: sample -> clip to [low, high] -> round.

    Ints are rounded to whole numbers; floats to FLOAT_DECIMALS places.
    """
    raw = _sample(distribution, rows, low, high, params or {})
    clipped = np.clip(raw, low, high)
    if kind == KIND_INT:
        return np.rint(clipped).astype(np.int64)
    return np.round(clipped, FLOAT_DECIMALS)


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
