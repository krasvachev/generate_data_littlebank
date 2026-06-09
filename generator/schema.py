"""Fixed schema for the littlebank synthetic dataset.

Column names, dtypes, categorical value lists, the unique-ID column, and
cross-column rules never change. Only the numeric ranges, the number of rows,
and the outcome class ratio are configurable at runtime (via the JSON config).

The column order below matches the reference output (df_sample.csv).
"""

# Generation "kinds" used to dispatch to the right generator.
KIND_ID = "id"                  # unique sequential integer
KIND_INT = "int"                # integer drawn from a numeric range
KIND_FLOAT = "float"            # float drawn from a numeric range
KIND_CATEGORICAL = "categorical"
KIND_BOOL = "bool"              # the target column (class-ratio controlled)

# Number of decimal places used for every float column.
FLOAT_DECIMALS = 4


class Column:
    """Definition of a single column in the fixed schema."""

    def __init__(self, name, kind, categories=None, default_range=None):
        self.name = name
        self.kind = kind
        self.categories = categories          # list[str] for categorical columns
        self.default_range = default_range    # (min, max) for numeric columns

    @property
    def is_numeric(self):
        return self.kind in (KIND_INT, KIND_FLOAT)


# Full, ordered schema. Numeric ranges here are defaults that the config may
# override. The config does not change names, dtypes or categories.
COLUMNS = [
    Column("contact_id", KIND_ID),
    Column("month", KIND_CATEGORICAL, categories=[
        "jan", "feb", "mar", "apr", "may", "jun",
        "jul", "aug", "sep", "oct", "nov", "dec",
    ]),
    Column("day_of_week", KIND_CATEGORICAL, categories=[
        "mon", "tue", "wed", "thu", "fri", "sat", "sun",
    ]),
    Column("contact", KIND_CATEGORICAL, categories=["mobile", "landline"]),
    Column("num_contacts", KIND_INT, default_range=(1, 10)),
    Column("days_since_previous", KIND_INT, default_range=(0, 30)),
    Column("num_contacts_previous", KIND_INT, default_range=(0, 10)),
    Column("outcome_previous", KIND_CATEGORICAL, categories=[
        "failure", "nonexistent", "success",
    ]),
    Column("age", KIND_INT, default_range=(16, 80)),
    Column("marital", KIND_CATEGORICAL, categories=[
        "married", "single", "divorced", "unknown",
    ]),
    Column("job", KIND_CATEGORICAL, categories=[
        "industrial", "unemployed", "administrative", "professional",
        "technical", "retired", "unknown", "entrepreneur", "management",
        "tradesman", "domestic", "full_time_education",
    ]),
    Column("education", KIND_CATEGORICAL, categories=[
        "4years", "university.degree", "high.school", "9years",
        "postgrad", "6years", "unknown", "none",
    ]),
    Column("default", KIND_CATEGORICAL, categories=["unknown", "no", "yes"]),
    Column("mortgage", KIND_CATEGORICAL, categories=["no", "yes", "unknown"]),
    Column("personal_loan", KIND_CATEGORICAL, categories=["no", "yes", "unknown"]),
    Column("call_centre_volume", KIND_FLOAT, default_range=(0.0, 15.0)),
    Column("high_temp", KIND_INT, default_range=(-2, 45)),
    Column("low_temp", KIND_INT, default_range=(-15, 20)),
    Column("forward_rate", KIND_FLOAT, default_range=(0.05, 5.05)),
    Column("num_employed", KIND_FLOAT, default_range=(23040.0, 24269.0)),
    Column("consumer_confidence", KIND_FLOAT, default_range=(-50.0, -26.0)),
    Column("price_index", KIND_FLOAT, default_range=(253.0, 261.0)),
    Column("employment_variation", KIND_FLOAT, default_range=(-3.04, 1.26)),
    Column("outcome", KIND_BOOL),
]

COLUMN_ORDER = [c.name for c in COLUMNS]
COLUMNS_BY_NAME = {c.name: c for c in COLUMNS}

# The unique sequential-ID column.
ID_COLUMN = "contact_id"

# The boolean target column whose True/False ratio is configurable.
TARGET_COLUMN = "outcome"

# Cross-column rules as (high, low) pairs requiring high > low on every row.
GREATER_THAN_RULES = [("high_temp", "low_temp")]
