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

## How values are generated

- **`contact_id`** — unique sequential integers starting at 1.
- **Categorical columns** — sampled uniformly from their fixed allowed values.
- **Numeric columns** — drawn from a configurable distribution (see
  [Distributions](#distributions)), clipped to `[min, max]`. Integers are
  rounded; floats are rounded to 4 decimal places.
- **`outcome`** (boolean target) — exactly `round(rows * outcome_true_ratio)`
  `True` values, shuffled across the rows.
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
