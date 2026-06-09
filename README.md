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
| `ranges` | Optional per-column numeric `{ "min", "max" }` overrides. Omit a column to use its schema default. |

Only numeric columns can appear under `ranges`. Names, types, and categorical
values are not configurable.

## How values are generated

- **`contact_id`** — unique sequential integers starting at 1.
- **Categorical columns** — sampled uniformly from their fixed allowed values.
- **Numeric columns** — drawn from a normal distribution centred on the
  midpoint of the range (std = range/6) and clipped to `[min, max]`. Integers
  are rounded; floats are rounded to 4 decimal places.
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
