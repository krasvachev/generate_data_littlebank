"""Tests validating the generated dataset against the fixed schema."""

import numpy as np
import pytest

from generator import generators as gen
from generator.cli import build_dataframe, resolve_distribution, validate_config
from generator.schema import (
    COLUMN_ORDER,
    COLUMNS,
    COLUMNS_BY_NAME,
    FLOAT_DECIMALS,
    GREATER_THAN_RULES,
    ID_COLUMN,
    KIND_CATEGORICAL,
    KIND_FLOAT,
    KIND_INT,
    TARGET_COLUMN,
)


def make_config(rows=2000, ratio=0.1):
    return {
        "rows": rows,
        "outcome_true_ratio": ratio,
        "ranges": {},  # use schema defaults
    }


@pytest.fixture(scope="module")
def df():
    np.random.seed(0)  # deterministic test data only
    return build_dataframe(make_config())


def test_columns_and_order(df):
    assert list(df.columns) == COLUMN_ORDER


def test_row_count(df):
    assert len(df) == 2000


def test_numeric_within_range(df):
    for col in COLUMNS:
        if col.kind in (KIND_INT, KIND_FLOAT):
            low, high = col.default_range
            assert df[col.name].min() >= low, col.name
            assert df[col.name].max() <= high, col.name


def test_integers_are_whole(df):
    for col in COLUMNS:
        if col.kind == KIND_INT:
            assert (df[col.name] == df[col.name].round()).all(), col.name


def test_floats_decimal_places(df):
    for col in COLUMNS:
        if col.kind == KIND_FLOAT:
            rounded = df[col.name].round(FLOAT_DECIMALS)
            assert (df[col.name] == rounded).all(), col.name


def test_contact_id_unique_sequential(df):
    ids = df[ID_COLUMN].tolist()
    assert ids == list(range(1, len(df) + 1))
    assert df[ID_COLUMN].is_unique


def test_categorical_values_valid(df):
    for col in COLUMNS:
        if col.kind == KIND_CATEGORICAL:
            assert set(df[col.name].unique()).issubset(set(col.categories)), col.name


def test_greater_than_rules(df):
    for high_name, low_name in GREATER_THAN_RULES:
        assert (df[high_name] > df[low_name]).all(), (high_name, low_name)


def test_outcome_ratio():
    np.random.seed(1)
    rows, ratio = 5000, 0.1
    df = build_dataframe(make_config(rows=rows, ratio=ratio))
    assert df[TARGET_COLUMN].dtype == bool
    assert int(df[TARGET_COLUMN].sum()) == round(rows * ratio)


def test_range_override():
    np.random.seed(2)
    config = {
        "rows": 1000,
        "outcome_true_ratio": 0.5,
        "ranges": {"age": {"min": 30, "max": 40}},
    }
    df = build_dataframe(config)
    assert df["age"].min() >= 30
    assert df["age"].max() <= 40


@pytest.mark.parametrize("distribution", sorted(gen.DISTRIBUTIONS))
def test_all_distributions_respect_range(distribution):
    """Every distribution still produces values within each column's range."""
    np.random.seed(3)
    config = {
        "rows": 3000,
        "outcome_true_ratio": 0.2,
        "distribution": distribution,
        "ranges": {},
    }
    df = build_dataframe(config)
    for col in COLUMNS:
        if col.kind in (KIND_INT, KIND_FLOAT):
            low, high = col.default_range
            assert df[col.name].min() >= low, (distribution, col.name)
            assert df[col.name].max() <= high, (distribution, col.name)


def test_discrete_distributions_are_integers_for_float_columns():
    """poisson/binomial yield whole numbers even before float rounding."""
    np.random.seed(4)
    for distribution in ("poisson", "binomial"):
        vals = gen.numeric(
            2000, 0.0, 15.0, KIND_FLOAT, distribution=distribution
        )
        assert (vals == np.round(vals)).all(), distribution


def test_per_column_distribution_overrides_global():
    config = {
        "rows": 10,
        "outcome_true_ratio": 0.5,
        "distribution": "normal",
        "ranges": {"age": {"min": 16, "max": 80, "distribution": "uniform"}},
    }
    dist, _ = resolve_distribution(COLUMNS_BY_NAME["age"], config)
    assert dist == "uniform"
    # A column without an override falls back to the global default.
    dist2, _ = resolve_distribution(COLUMNS_BY_NAME["call_centre_volume"], config)
    assert dist2 == "normal"


def test_random_alias_matches_uniform_bounds():
    """'random' (absolute random) stays within range like uniform."""
    np.random.seed(5)
    vals = gen.numeric(5000, -3.04, 1.26, KIND_FLOAT, distribution="random")
    assert vals.min() >= -3.04
    assert vals.max() <= 1.26


def test_invalid_distribution_rejected():
    with pytest.raises(ValueError):
        validate_config({
            "rows": 10,
            "outcome_true_ratio": 0.1,
            "distribution": "not_a_distribution",
        })
    with pytest.raises(ValueError):
        validate_config({
            "rows": 10,
            "outcome_true_ratio": 0.1,
            "ranges": {"age": {"min": 16, "max": 80, "distribution": "bogus"}},
        })


# --- Dependent outcome target (outcome_model) ----------------------------------

def make_model_config(rows=5000, ratio=0.1, weights=None):
    config = make_config(rows=rows, ratio=ratio)
    if weights is not None:
        config["outcome_model"] = {"weights": weights}
    return config


def test_outcome_model_preserves_exact_count():
    np.random.seed(10)
    rows, ratio = 5000, 0.1
    df = build_dataframe(make_model_config(
        rows=rows, ratio=ratio,
        weights={"contact": {"mobile": 5.0, "landline": 1.0}},
    ))
    assert df[TARGET_COLUMN].dtype == bool
    assert int(df[TARGET_COLUMN].sum()) == round(rows * ratio)


def test_outcome_model_creates_dependence():
    """A high-weight category gets a higher realized True rate."""
    np.random.seed(11)
    df = build_dataframe(make_model_config(
        weights={"contact": {"mobile": 5.0, "landline": 1.0}},
    ))
    mobile_rate = df.loc[df["contact"] == "mobile", TARGET_COLUMN].mean()
    landline_rate = df.loc[df["contact"] == "landline", TARGET_COLUMN].mean()
    assert mobile_rate > landline_rate


def test_outcome_model_combines_multiplicatively():
    """Weights from two driver columns compound (product)."""
    np.random.seed(12)
    df = build_dataframe(make_model_config(
        rows=8000, ratio=0.25,
        weights={
            "contact": {"mobile": 4.0, "landline": 1.0},
            "month": {"may": 4.0},
        },
    ))
    best = df[(df["contact"] == "mobile") & (df["month"] == "may")]
    worst = df[(df["contact"] == "landline") & (df["month"] != "may")]
    assert best[TARGET_COLUMN].mean() > worst[TARGET_COLUMN].mean()


def test_outcome_model_equal_weights_match_uniform_count():
    """Equal weights still yield the exact count (reduces to uniform)."""
    np.random.seed(13)
    rows, ratio = 4000, 0.2
    df = build_dataframe(make_model_config(
        rows=rows, ratio=ratio,
        weights={"contact": {"mobile": 1.0, "landline": 1.0}},
    ))
    assert int(df[TARGET_COLUMN].sum()) == round(rows * ratio)


def test_boolean_dependent_extremes():
    weights = np.ones(100)
    assert gen.boolean_dependent(100, 0.0, weights).sum() == 0
    assert gen.boolean_dependent(100, 1.0, weights).sum() == 100


def test_outcome_model_validation_rejects_bad_config():
    base = {"rows": 10, "outcome_true_ratio": 0.1}
    # Unknown column.
    with pytest.raises(ValueError):
        validate_config({**base, "outcome_model": {"weights": {"nope": {}}}})
    # Non-categorical column.
    with pytest.raises(ValueError):
        validate_config({**base, "outcome_model": {"weights": {"age": {"1": 2.0}}}})
    # Unknown category value.
    with pytest.raises(ValueError):
        validate_config({
            **base, "outcome_model": {"weights": {"contact": {"fax": 2.0}}},
        })
    # Negative weight.
    with pytest.raises(ValueError):
        validate_config({
            **base, "outcome_model": {"weights": {"contact": {"mobile": -1.0}}},
        })
    # weights must be an object.
    with pytest.raises(ValueError):
        validate_config({**base, "outcome_model": {"weights": []}})


def test_outcome_model_valid_config_passes():
    validate_config({
        "rows": 10,
        "outcome_true_ratio": 0.1,
        "outcome_model": {"weights": {"contact": {"mobile": 2.0}}},
    })
