# LittleBank - Synthetic Data Generator

**A small, configurable Python tool that generates synthetic, bank-marketing-style
CSV datasets**. The dataset has a **fixed, schema-validated layout** (24 columns). Number of rows, numeric ranges, statistical distributions and the class balance of the
target are all configurable per run via a single JSON file. On the other side, column names,
data types, and categorical values are pinned in code so every generated file is
schema-compatible with the next.

The schema is modeled on the public
[Bank Marketing dataset](https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing/data)
(client demographics, contact history, macroeconomic indicators, and a binary
`outcome` target), with a few tweaks for this project.

## Why this project exists

This generator is the **upstream data source** for my end-to-end data science project:

> 📊 **[Data-Science-Case-Study](https://github.com/krasvachev/Data-Science-Case-Study)** - exploratory analysis, feature engineering and predictive modeling on the `LittleBank` dataset.

Working with synthetic data keeps the two repositories cleanly separated and
gives me full control over the experiment:

- **Reproducible inputs** - regenerate the exact same shape of data on demand, without redistributing a third-party dataset.
- **Controlled experiments** - dial the row count, per-column distributions, and
  the `outcome` class ratio up or down to stress-test models against class
  imbalance, skew and scale.
- **Stable contract** - the fixed schema guarantees that every regenerated CSV
  drops straight into the downstream analysis pipeline.

## Features

- **Fixed 24-column schema** - names, dtypes, and categorical values are defined
  once in code and never drift between runs.
- **Six selectable distributions** for every numeric *and* categorical column
  (`normal`, `uniform`, `exponential`, `poisson`, `binomial`, `random`).
- **Global + per-column configuration** - set a default distribution for all
  columns and override individual ones.
- **Exact target class balance** - the boolean `outcome` column gets precisely
  `round(rows × ratio)` `True` values.
- **Optional feature-dependent target** - bias `outcome` toward chosen feature
  values so the data carries real signal, while keeping the exact class balance.
- **Built-in integrity rules** - unique sequential IDs and an enforced
  `high_temp > low_temp` cross-column constraint.
- **Validated config** - invalid distributions, unknown columns, and bad ranges
  fail fast with clear error messages.
- **Tested** - a `pytest` suite verifies schema, ranges, formatting, and rules.

## Requirements

- Python 3.11+
- [pandas](https://pandas.pydata.org/) ≥ 2.2 and [NumPy](https://numpy.org/) ≥ 1.26

## Installation

The first step is to clone the Repo and create a virtual environment.

```bash
git clone https://github.com/krasvachev/generate_data_littlebank.git
cd generate_data_littlebank   # The path where you clone the Repo         
```

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
python --version              
```

Afterwards, you should install the requirements.txt and then run the generator:

```bash
pip install -r requirements.txt
python -m generator.cli --config config.example.json
```

## Usage

Generate a dataset from a config file:

```bash
python -m generator.cli --config config.example.json
```

This writes a dated CSV - e.g. `output/littlebank_2026-06-10.csv`. The file is
comma-delimited, UTF-8 encoded, has a header row and no index column. On
completion the tool prints the row count and the `outcome` class split:

```
Wrote 1000 rows to output/littlebank_2026-06-10.csv (100 True / 900 False on outcome)
```

## Configuration

Runs are driven entirely by a JSON config (see `config.example.json`). Only the
keys below are configurable - column names, data types and the set of allowed
categorical values are fixed in the schema and cannot be changed at runtime.

| Key | Required | Meaning |
|-----|----------|---------|
| `rows` | ✅ | Number of rows to generate (positive integer). |
| `outcome_true_ratio` | ✅ | Fraction of `outcome` values that are `True`, e.g. `0.1` → 10% `True`, 90% `False`. |
| `output_base` | - | Output path prefix; the run date and `.csv` are appended. Defaults to `output/littlebank`. |
| `distribution` | - | Global default distribution for all numeric columns. Defaults to `normal`. |
| `params` | - | Global default distribution parameters for numeric columns. |
| `ranges` | - | Per-column numeric overrides: `{ "min", "max", "distribution", "params" }`. Omit a column to use its schema default. |
| `categorical_distribution` | - | Global default distribution for all categorical columns. Defaults to `uniform`. |
| `categorical_params` | - | Global default distribution parameters for categorical columns. |
| `categorical` | - | Per-column categorical overrides: `{ "distribution", "params" }`, keyed by column name. |
| `outcome_model` | - | Makes `outcome` depend on feature columns instead of being uniformly random (see [Dependent outcome target](#dependent-outcome-target)). Omit for the default behaviour. |

A companion `config.help.json` documents every key and the available
distributions. It is **reference material only** - the generator never reads it.

### Distributions

Each column is drawn from one of six selectable distributions. A distribution can
be set **globally** and/or **per column**; the per-column setting overrides the
global one, which overrides the built-in default. Whatever the distribution,
numeric values are always clipped to the column's `[min, max]` and rounded to its
type (ints to whole numbers, floats to 4 decimals).

| Name | Shape | Default parameters | Optional `params` |
|------|-------|--------------------|-------------------|
| `normal` | Gaussian, centred on the range midpoint | `std = range/6` | `mean`, `std` |
| `uniform` | Flat across `[min, max]` | - | - |
| `exponential` | Right-skewed, anchored at `min` | `scale = range/3` | `scale` |
| `poisson` | Discrete counts, anchored at `min` | `lam = range/2` | `lam` |
| `binomial` | Discrete, anchored at `min` | `n = round(range)`, `p = 0.5` | `n`, `p` |
| `random` | Uniform within range (the "absolute random" mode; alias of `uniform`) | - | - |

**Numeric example** - a global default with per-column overrides:

```json
{
  "rows": 1000,
  "outcome_true_ratio": 0.1,
  "distribution": "normal",
  "ranges": {
    "age": {"min": 16, "max": 80, "distribution": "normal"},
    "call_centre_volume": {"min": 0.0, "max": 15.0, "distribution": "exponential", "params": {"scale": 3.0}},
    "days_since_previous": {"min": 0, "max": 30, "distribution": "poisson"},
    "num_employed": {"min": 23040.0, "max": 24269.0, "distribution": "random"}
  }
}
```

#### Categorical columns

The same six distributions apply to categorical columns. The distribution is
mapped onto the **category order** (index `0 … k-1` as declared in the schema):
`normal` favours the middle categories, `exponential`/`poisson` favour the first
ones, `binomial` centres them, and `uniform`/`random` weight every category
equally (the default).

```json
{
  "rows": 1000,
  "outcome_true_ratio": 0.1,
  "categorical_distribution": "uniform",
  "categorical": {
    "month": {"distribution": "normal"},
    "job": {"distribution": "exponential"}
  }
}
```

### Dependent outcome target

By default `outcome` is uniformly random - it carries no signal about any feature
column, so a model trained on the data has nothing to learn. Add an
`outcome_model` block to make `outcome` **to depends** on one or more categorical
columns while keeping the exact `True` count.

```json
{
  "rows": 1000,
  "outcome_true_ratio": 0.1,
  "outcome_model": {
    "weights": {
      "contact": {"mobile": 2.0, "landline": 1.0},
      "month":   {"may": 3.0, "jun": 2.0, "dec": 0.5}
    }
  }
}
```

- Each driver column maps categories to **relative weights** (propensities, not
  literal rates). Categories you don't list default to weight `1.0`.
- A row's selection weight is the **product** of its drivers' weights. Above, a
  `contact=mobile, month=may` row has weight `2.0 × 3.0 = 6.0`, while a
  `contact=landline, month=dec` row has `1.0 × 0.5 = 0.5` - making it 12× less
  likely to be `True`.
- The number of `True` values is still **exactly**
  `round(rows × outcome_true_ratio)`; the weights only steer *which* rows are
  chosen (weighted sampling without replacement). Equal weights everywhere
  reproduce the default uniform behaviour.

Only **categorical** columns may be drivers, listed categories must exist in the
schema, and weights must be non-negative numbers.

## How values are generated

- **`contact_id`** - unique sequential integers starting at 1.
- **Categorical columns** - sampled from their fixed allowed values, uniformly by
  default or shaped by a configurable distribution (see
  [Categorical columns](#categorical-columns)).
- **Numeric columns** - drawn from a configurable distribution (see
  [Distributions](#distributions)), clipped to `[min, max]`. Integers are rounded
  to whole numbers; floats are rounded to 4 decimal places.
- **`outcome`** (boolean target) - exactly `round(rows × outcome_true_ratio)`
  `True` values. Shuffled uniformly across the rows by default, or biased toward
  high-weight rows when an `outcome_model` is configured (see
  [Dependent outcome target](#dependent-outcome-target)).
- **Cross-column rule** - `high_temp > low_temp` is enforced on every row.

## Reproducibility

Sampling uses NumPy's global random state. The CLI does **not** seed it, so each
run produces a fresh, independent dataset. The dated output filename keeps
successive runs from overwriting one another.

If you need byte-for-byte reproducible output (e.g. to pin the dataset behind the
downstream case study), seed NumPy before generating. The test suite does exactly
this with `np.random.seed(...)`, and you can do the same from a short script:

```python
import numpy as np
from generator.cli import build_dataframe

np.random.seed(42)  # any fixed seed
df = build_dataframe({"rows": 1000, "outcome_true_ratio": 0.1, "ranges": {}})
df.to_csv("output/littlebank_seeded.csv", index=False)
```

Given the same seed, NumPy version, and config, this yields identical data on
every run.

## Project structure

```
generate_data_littlebank/
├── generator/
│   ├── __init__.py
│   ├── cli.py            # entry point: load config → validate → build → write CSV
│   ├── schema.py         # fixed 24-column schema (names, types, categories, rules)
│   └── generators.py     # distribution samplers and per-column value generators
├── tests/
│   └── test_generator.py # schema, range, formatting, distribution, and rule checks
├── config.example.json   # runnable example configuration
├── config.help.json      # human-readable documentation of every config key
├── requirements.txt
├── LICENSE
└── README.md
```

## Tests

```bash
pytest
```

The suite verifies column order, row count, value ranges, integer/float
formatting, `contact_id` uniqueness, valid categorical values, the
`high_temp > low_temp` rule, the `outcome` class ratio, that every distribution
(numeric and categorical) stays within its allowed values. The optional
`outcome_model` biases the target while preserving the exact class balance.

## License

Released under the [MIT License](LICENSE). © 2026 Krasimir Vachev.

## Author

**Krasimir Vachev** - [@krasvachev](https://github.com/krasvachev)
