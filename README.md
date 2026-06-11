# littlebank synthetic data generator

Generates a CSV of synthetic records matching a **fixed schema** (24 columns).
Column names, data types, and categorical values are fixed in code; the number
of rows, the numeric ranges, and the True/False ratio of the `outcome` target
are configurable per run via a JSON config.

## Install

```bash
pip install -r requirements.txt
```

## Usage

```bash
python -m generator.cli --config config.example.json
```

This writes a dated CSV, e.g. `output/littlebank_2026-06-09.csv`
(comma-delimited, header row, UTF-8, no index column).

## Configuration (`config.example.json`)

| Key | Meaning |
|-----|---------|
| `rows` | Number of rows to generate (positive integer). |
| `outcome_true_ratio` | Fraction of `outcome` values that are `True` (0–1). e.g. `0.1` → 10% True, 90% False. |
| `output_base` | Output path prefix; the run date and `.csv` are appended. |
| `distribution` | Global default distribution for all numeric columns (see below). |
| `params` | Optional global distribution parameters (see below). |
| `ranges` | Optional per-column numeric `{ "min", "max", "distribution", "params" }` overrides. Omit a column to use its schema default. |
| `outcome_model` | Optional. Makes `outcome` depend on feature columns (see [Dependent outcome target](#dependent-outcome-target)). Omit for the default uniform behaviour. |

Only numeric columns can appear under `ranges`. Names, types, and categorical
values are not configurable.

## Distributions

Each numeric column is drawn from one of six selectable distributions. A
distribution can be set **globally** (top-level `distribution`) and/or
**per-column** (inside a `ranges` entry); a per-column setting overrides the
global one, which overrides the built-in default (`normal`). Whatever the
distribution, values are always clipped to the column's `[min, max]` and
rounded to the column's type (ints to whole numbers, floats to 4 decimals).

| Name | Shape | Default parameters | Optional `params` |
|------|-------|--------------------|-------------------|
| `normal` | Gaussian, centred on the range midpoint | `std = range/6` | `mean`, `std` |
| `uniform` | Flat across `[min, max]` | — | — |
| `exponential` | Right-skewed, anchored at `min` | `scale = range/3` | `scale` |
| `poisson` | Discrete counts, anchored at `min` | `lam = range/2` | `lam` |
| `binomial` | Discrete, anchored at `min` | `n = round(range)`, `p = 0.5` | `n`, `p` |
| `random` | Uniform within range (the "absolute random" mode; alias of `uniform`) | — | — |

Example — global default with per-column overrides:

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

## Dependent outcome target

By default `outcome` is uniformly random — it carries no signal about any
feature column, so a model trained on the data finds nothing to learn. Add an
`outcome_model` block to make `outcome` **depend** on one or more categorical
columns while keeping the exact True count.

```json
"outcome_model": {
  "weights": {
    "contact": {"mobile": 2.0, "landline": 1.0},
    "month":   {"may": 3.0, "jun": 2.0, "dec": 0.5}
  }
}
```

- Each driver column maps categories to **relative weights** (propensities, not
  literal rates). Categories you don't list default to weight `1.0`.
- A row's selection weight is the **product** of its drivers' weights. With the
  example above, a `contact=mobile, month=may` row has weight `2.0 × 3.0 = 6.0`,
  while a `contact=landline, month=dec` row has `1.0 × 0.5 = 0.5` — making it
  12× less likely to be `True`.
- The total number of `True` values is still **exactly**
  `round(rows * outcome_true_ratio)`; the weights only steer *which* rows are
  chosen (weighted sampling without replacement). Equal weights everywhere
  reproduce the default uniform behaviour.

Only **categorical** columns may be used as drivers, and listed categories must
exist in the schema. Weights must be non-negative numbers.

## How values are generated

- **`contact_id`** — unique sequential integers starting at 1.
- **Categorical columns** — sampled uniformly from their fixed allowed values.
- **Numeric columns** — drawn from a configurable distribution (see
  [Distributions](#distributions)), clipped to `[min, max]`. Integers are
  rounded; floats are rounded to 4 decimal places.
- **`outcome`** (boolean target) — exactly `round(rows * outcome_true_ratio)`
  `True` values. By default these are shuffled uniformly across the rows; with
  an `outcome_model` they are biased toward high-weight rows (see
  [Dependent outcome target](#dependent-outcome-target)).
- **Cross-column rule** — `high_temp > low_temp` is enforced on every row.

## Schema

See `generator/schema.py` for the authoritative column definitions (names,
types, categories, ranges, unique-ID column, and cross-column rules).

## Tests

```bash
pytest
```

Tests verify column order, row count, value ranges, integer/float formatting,
`contact_id` uniqueness, valid categorical values, the `high_temp > low_temp`
rule, and the `outcome` class ratio.
