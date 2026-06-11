"""Per-column value generators.

Numeric columns are drawn from one of six selectable distributions, then
clipped to the column's range and rounded according to its type (ints to whole
numbers, floats to FLOAT_DECIMALS places). Categorical columns also can be sampled
from six selectable distributions. The unique-ID column is sequential, and the boolean target is
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

DEFAULT_DISTRIBUTION = NORMAL


def sequential_ids(rows: int, start: int = 1) -> np.ndarray:
    """Unique sequential integer IDs: start, start+1, ..."""
    return np.arange(start, start + rows, dtype=np.int64)


# --- distribution samplers ---------------------------------------------------
# Each sampler draws raw (unclipped, unrounded) values for ``rows`` draws over
# the range [low, high], honouring optional per-column ``params``. The
# DISTRIBUTIONS set is derived from this registry, so the valid names and the
# implementations can never drift apart.

def _sample_normal(rows: int, low: float, high: float, params: dict) -> np.ndarray:
    mean = params.get("mean", (low + high) / 2.0)
    std = params.get("std", (high - low) / 6.0)
    if std == 0:
        return np.full(rows, mean, dtype=float)
    return np.random.normal(loc=mean, scale=std, size=rows)


def _sample_uniform(rows: int, low: float, high: float, params: dict) -> np.ndarray:
    return np.random.uniform(low, high, size=rows)


def _sample_exponential(rows: int, low: float, high: float, params: dict) -> np.ndarray:
    span = high - low
    scale = params.get("scale", span / 3.0 if span else 1.0)
    return low + np.random.exponential(scale=scale, size=rows)


def _sample_poisson(rows: int, low: float, high: float, params: dict) -> np.ndarray:
    lam = params.get("lam", (high - low) / 2.0)
    return low + np.random.poisson(lam=lam, size=rows)


def _sample_binomial(rows: int, low: float, high: float, params: dict) -> np.ndarray:
    n = params.get("n", max(int(round(high - low)), 1))
    p = params.get("p", 0.5)
    return low + np.random.binomial(n=n, p=p, size=rows)


_SAMPLERS = {
    NORMAL: _sample_normal,
    UNIFORM: _sample_uniform,
    RANDOM: _sample_uniform,  # absolute random == uniform within range
    EXPONENTIAL: _sample_exponential,
    POISSON: _sample_poisson,
    BINOMIAL: _sample_binomial,
}

# Valid distribution names, derived from the sampler registry above.
DISTRIBUTIONS = set(_SAMPLERS)


def _sample(distribution: str, rows: int, low: float, high: float,
            params: dict) -> np.ndarray:
    """Draw raw (unclipped, unrounded) values for the given distribution."""
    try:
        sampler = _SAMPLERS[distribution]
    except KeyError:
        raise ValueError(f"unknown distribution: {distribution!r}")
    return sampler(rows, low, high, params)


def numeric(rows: int, low: float, high: float, kind: str,
            distribution: str = DEFAULT_DISTRIBUTION,
            params: dict | None = None) -> np.ndarray:
    """Generate a numeric column: sample -> clip to [low, high] -> round.

    Ints are rounded to whole numbers; floats to FLOAT_DECIMALS places.
    """
    raw = _sample(distribution, rows, low, high, params or {})
    clipped = np.clip(raw, low, high)
    if kind == KIND_INT:
        return np.rint(clipped).astype(np.int64)
    return np.round(clipped, FLOAT_DECIMALS)


def categorical(rows: int, categories: list, distribution: str = UNIFORM,
                params: dict | None = None) -> np.ndarray:
    """Sample categorical values, optionally shaped by a distribution.

    With 'uniform'/'random' every category is equally likely. For the other
    distributions, the distribution is mapped onto the category index range
    [0, n_categories-1] using the same numeric generator, so e.g. 'normal'
    favours the middle categories and 'exponential' favours the first ones.
    Category order is the order declared in the schema.
    """
    category_values = np.asarray(categories, dtype=object)
    n_categories = len(category_values)
    if n_categories <= 1 or distribution in (UNIFORM, RANDOM):
        return np.random.choice(category_values, size=rows)
    idx = numeric(rows, 0, n_categories - 1, KIND_INT, distribution, params or {})
    return category_values[idx]


def boolean_by_ratio(rows: int, true_ratio: float) -> np.ndarray:
    """Boolean array with round(rows * true_ratio) Trues, shuffled."""
    n_true = int(round(rows * true_ratio))
    values = np.zeros(rows, dtype=bool)
    values[:n_true] = True
    np.random.shuffle(values)
    return values


def boolean_dependent(rows: int, true_ratio: float,
                      weights: np.ndarray) -> np.ndarray:
    """Boolean target whose Trues are biased toward high-weight rows.

    Produces exactly round(rows * true_ratio) Trues (same guarantee as
    boolean_by_ratio), but instead of scattering them uniformly the rows are
    selected by weighted sampling without replacement: each row i is assigned a
    key e_i / weights[i] with e_i ~ Exp(1), and the n_true rows with the
    smallest keys become True. A row's chance of selection is proportional to
    its weight, so larger weights mean a higher True rate. With all weights
    equal this reduces to uniform selection (i.e. boolean_by_ratio).
    """
    n_true = int(round(rows * true_ratio))
    values = np.zeros(rows, dtype=bool)
    if n_true <= 0:
        return values
    if n_true >= rows:
        values[:] = True
        return values
    weights = np.asarray(weights, dtype=float)
    # Exponential keys divided by weight: smallest keys win, so higher weight
    # means a smaller expected key and a better chance of selection. Zero-weight
    # rows get an infinite key (never chosen unless n_true forces it).
    with np.errstate(divide="ignore"):
        keys = np.random.exponential(size=rows) / weights
    chosen = np.argpartition(keys, n_true)[:n_true]
    values[chosen] = True
    return values


def enforce_greater_than(high_vals: np.ndarray, low_vals: np.ndarray,
                         high_low: int, high_high: int) -> np.ndarray:
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
            np.random.randint(lower, high_high + 1)
            for lower in lower_bounds
        ], dtype=high_vals.dtype)
    return high_vals
