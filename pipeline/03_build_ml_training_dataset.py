import os
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

INPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "hk_transactions_with_canonical_v2.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs"
)

SHOPPING_VALIDATION_FILE = os.path.join(
    OUTPUT_DIR,
    "shopping_candidate_validation.csv"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "ml_training_dataset.csv"
)

SUMMARY_FILE = os.path.join(
    OUTPUT_DIR,
    "ml_training_dataset_summary.csv"
)


# ============================================================
# CONSTANTS
# ============================================================

SHOPPING_CANONICAL_ID = 2006
SHOPPING_FAMILY = "Shopping & Retail"

VALID_SHOPPING_STATUS = "STRONG_SHOPPING"

# Maximum number of training examples per category.
#
# This prevents Shopping from dominating the model while
# preserving all available examples for smaller categories.
MAX_PER_CATEGORY = 2000


# ============================================================
# CANONICAL TAXONOMY
# ============================================================

CANONICAL_TAXONOMY = {
    2001: "Food & Groceries",
    2002: "Fuel",
    2003: "Transport",
    2004: "Utilities & Bills",
    2005: "Healthcare",
    2006: "Shopping & Retail",
    2007: "Transfers & Withdrawals",
    2008: "Income",
    2009: "Other",
}


# ============================================================
# FAMILY NORMALIZATION
# ============================================================

FAMILY_ALIASES = {

    # Food
    "food": "Food & Groceries",
    "food & groceries": "Food & Groceries",
    "groceries": "Food & Groceries",
    "grocery": "Food & Groceries",
    "dining": "Food & Groceries",

    # Fuel
    "fuel": "Fuel",

    # Transport
    "transport": "Transport",

    # Utilities
    "utilities": "Utilities & Bills",
    "utilities & bills": "Utilities & Bills",

    # Healthcare
    "healthcare": "Healthcare",

    # Shopping
    "shopping": "Shopping & Retail",
    "shopping & retail": "Shopping & Retail",

    # Transfers
    "transfers": "Transfers & Withdrawals",
    "transfers & withdrawals": "Transfers & Withdrawals",

    # Income
    "income": "Income",

    # Other
    "other": "Other",
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_text(value):
    """Safely convert a value to normalized text."""

    if pd.isna(value):
        return ""

    return str(value).strip()


def normalize_description(value):
    """Normalize transaction descriptions."""

    text = clean_text(value).lower()

    text = " ".join(text.split())

    return text


def normalize_family(value):
    """Convert old/alternative family names into the canonical taxonomy."""

    if pd.isna(value):
        return ""

    text = str(value).strip().lower()

    return FAMILY_ALIASES.get(
        text,
        str(value).strip()
    )


def first_existing_column(df, candidates):
    """Return the first candidate column that exists."""

    for column in candidates:

        if column in df.columns:
            return column

    return None


# ============================================================
# START
# ============================================================

print("=" * 90)
print("BUILDING EXPANDED TRUSTED ML TRAINING DATASET")
print("=" * 90)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD MAIN DATASET
# ============================================================

print()
print("Loading canonical transaction dataset:")
print(INPUT_PATH)

if not os.path.exists(INPUT_PATH):

    raise FileNotFoundError(
        f"Input dataset not found:\n{INPUT_PATH}"
    )


df = pd.read_csv(
    INPUT_PATH
)

print(
    f"Transactions loaded: {len(df):,}"
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "vch_desc",
    "vch_type",
    "canonical_id",
    "canonical_family",
    "mapping_method",
]


missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing_columns:

    raise ValueError(
        "Missing required columns in input dataset:\n"
        + "\n".join(
            f"- {column}"
            for column in missing_columns
        )
    )


# ============================================================
# LOAD SHOPPING VALIDATION
# ============================================================

print()
print("=" * 90)
print("LOADING VALIDATED SHOPPING LABELS")
print("=" * 90)


if not os.path.exists(
    SHOPPING_VALIDATION_FILE
):

    raise FileNotFoundError(
        f"Shopping validation file not found:\n"
        f"{SHOPPING_VALIDATION_FILE}"
    )


shopping_validation = pd.read_csv(
    SHOPPING_VALIDATION_FILE
)


print(
    f"Shopping validation records: "
    f"{len(shopping_validation):,}"
)


# ============================================================
# FIND SHOPPING VALIDATION COLUMNS
# ============================================================

validation_column = first_existing_column(
    shopping_validation,
    [
        "shopping_validation",
        "validation_status",
        "shopping_decision",
    ]
)


if validation_column is None:

    raise ValueError(
        "Could not find Shopping validation-status column."
    )


shopping_desc_column = first_existing_column(
    shopping_validation,
    [
        "vch_desc",
        "ml_description",
        "description",
    ]
)


if shopping_desc_column is None:

    raise ValueError(
        "Could not find transaction description column "
        "in Shopping validation dataset."
    )


# ============================================================
# VALIDATED SHOPPING LABELS
# ============================================================

print()
print("=" * 90)
print("VALIDATING SHOPPING LABELS")
print("=" * 90)


shopping_validation["_normalized_desc"] = (
    shopping_validation[
        shopping_desc_column
    ].apply(
        normalize_description
    )
)


shopping_validation["_validation_status"] = (
    shopping_validation[
        validation_column
    ]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.upper()
)


strong_shopping = shopping_validation[
    shopping_validation[
        "_validation_status"
    ]
    == VALID_SHOPPING_STATUS
].copy()


print(
    f"Strong Shopping candidates: "
    f"{len(strong_shopping):,}"
)


# ============================================================
# SOURCE 1:
# APPROVED MAPPINGS ONLY
# ============================================================

print()
print("=" * 90)
print("SOURCE 1 - APPROVED MAPPINGS")
print("=" * 90)


approved = df[
    df["mapping_method"]
    .astype(str)
    .str.strip()
    .str.upper()
    == "APPROVED_MAPPING"
].copy()


print(
    f"Approved mapping records: "
    f"{len(approved):,}"
)


# ============================================================
# SOURCE 2:
# DETERMINISTIC BUSINESS RULES ONLY
# ============================================================

print()
print("=" * 90)
print("SOURCE 2 - DETERMINISTIC RULE LABELS")
print("=" * 90)


rule_labels = df[
    df["mapping_method"]
    .astype(str)
    .str.strip()
    .str.upper()
    == "RULE"
].copy()


print(
    f"Deterministic rule records: "
    f"{len(rule_labels):,}"
)


# ============================================================
# IMPORTANT:
# ML-GENERATED LABELS ARE EXPLICITLY EXCLUDED
# ============================================================

ml_labels = df[
    df["mapping_method"]
    .astype(str)
    .str.strip()
    .str.upper()
    == "ML"
].copy()


print()
print(
    f"ML-generated labels excluded: "
    f"{len(ml_labels):,}"
)


# ============================================================
# PREPARE TRUSTED LABEL DATA
# ============================================================

trusted_source_columns = [
    "vch_desc",
    "vch_type",
    "canonical_id",
    "canonical_family",
]


def prepare_trusted_source(
    source_df,
    source_name,
    confidence
):

    if len(source_df) == 0:

        return pd.DataFrame(
            columns=[
                "vch_desc",
                "vch_type",
                "canonical_id",
                "canonical_family",
                "label_source",
                "label_confidence",
                "validation_status",
            ]
        )

    result = source_df[
        trusted_source_columns
    ].copy()


    # Normalize family names.

    result["canonical_family"] = (
        result[
            "canonical_family"
        ].apply(
            normalize_family
        )
    )


    # Convert canonical IDs to numeric.

    result["canonical_id"] = pd.to_numeric(
        result["canonical_id"],
        errors="coerce"
    )


    # Keep only valid canonical IDs.

    result = result[
        result["canonical_id"].isin(
            CANONICAL_TAXONOMY.keys()
        )
    ].copy()


    # Ensure the family matches our taxonomy.

    result = result[
        result.apply(
            lambda row:
                CANONICAL_TAXONOMY.get(
                    int(row["canonical_id"])
                ) == row["canonical_family"],
            axis=1
        )
    ].copy()


    # Remove empty descriptions.

    result["_normalized_desc"] = (
        result["vch_desc"].apply(
            normalize_description
        )
    )


    result = result[
        result["_normalized_desc"] != ""
    ].copy()


    result["label_source"] = source_name

    result["label_confidence"] = confidence

    result["validation_status"] = source_name


    return result[
        [
            "vch_desc",
            "vch_type",
            "canonical_id",
            "canonical_family",
            "label_source",
            "label_confidence",
            "validation_status",
            "_normalized_desc",
        ]
    ].copy()


approved_training = prepare_trusted_source(
    approved,
    "approved_mapping",
    "very_high"
)


rule_training = prepare_trusted_source(
    rule_labels,
    "deterministic_rule",
    "high"
)


print()
print(
    f"Usable approved labels: "
    f"{len(approved_training):,}"
)

print(
    f"Usable rule labels: "
    f"{len(rule_training):,}"
)


# ============================================================
# PREPARE HUMAN-VALIDATED SHOPPING
# ============================================================

shopping_training = pd.DataFrame(
    columns=[
        "vch_desc",
        "vch_type",
        "canonical_id",
        "canonical_family",
        "label_source",
        "label_confidence",
        "validation_status",
        "_normalized_desc",
    ]
)


if len(strong_shopping) > 0:

    shopping_training = pd.DataFrame({

        "vch_desc":
            strong_shopping[
                shopping_desc_column
            ].astype(str),

        "vch_type":
            "expense",

        "canonical_id":
            SHOPPING_CANONICAL_ID,

        "canonical_family":
            SHOPPING_FAMILY,

        "label_source":
            "human_validated_shopping",

        "label_confidence":
            "very_high",

        "validation_status":
            VALID_SHOPPING_STATUS,

        "_normalized_desc":
            strong_shopping[
                "_normalized_desc"
            ],
    })


print()
print(
    f"Human-validated Shopping labels: "
    f"{len(shopping_training):,}"
)


# ============================================================
# COMBINE TRUSTED SOURCES
# ============================================================

print()
print("=" * 90)
print("COMBINING TRUSTED LABEL SOURCES")
print("=" * 90)


training_parts = []


if len(approved_training) > 0:

    training_parts.append(
        approved_training
    )


if len(rule_training) > 0:

    training_parts.append(
        rule_training
    )


if len(shopping_training) > 0:

    training_parts.append(
        shopping_training
    )


if not training_parts:

    raise ValueError(
        "No trusted training records were produced."
    )


training = pd.concat(
    training_parts,
    ignore_index=True
)


print(
    f"Combined records before deduplication: "
    f"{len(training):,}"
)


# ============================================================
# REMOVE CONFLICTING DESCRIPTIONS
# ============================================================

print()
print("=" * 90)
print("CHECKING LABEL CONFLICTS")
print("=" * 90)


conflict_counts = (
    training
    .groupby(
        "_normalized_desc"
    )[
        "canonical_id"
    ]
    .nunique()
)


conflicting_descriptions = set(
    conflict_counts[
        conflict_counts > 1
    ].index
)


print(
    f"Conflicting descriptions found: "
    f"{len(conflicting_descriptions):,}"
)


if conflicting_descriptions:

    training = training[
        ~training[
            "_normalized_desc"
        ].isin(
            conflicting_descriptions
        )
    ].copy()


    print(
        "Conflicting descriptions removed."
    )


# ============================================================
# REMOVE DUPLICATES
# ============================================================

print()
print("=" * 90)
print("REMOVING DUPLICATES")
print("=" * 90)


before_dedup = len(training)


# We deduplicate by description + category.
#
# This means repeated transactions with exactly the same
# description do not artificially inflate the training set.

training = training.drop_duplicates(
    subset=[
        "_normalized_desc",
        "canonical_id",
    ]
).copy()


print(
    f"Duplicates removed: "
    f"{before_dedup - len(training):,}"
)


# ============================================================
# LIMIT DOMINANT CATEGORIES
# ============================================================

print()
print("=" * 90)
print("BALANCING TRAINING DATASET")
print("=" * 90)


before_balancing = len(training)


balanced_parts = []


for canonical_id, category_name in CANONICAL_TAXONOMY.items():

    category_data = training[
        training[
            "canonical_id"
        ] == canonical_id
    ].copy()


    if len(category_data) == 0:

        continue


    # Keep every record if category is below
    # the maximum.

    if len(category_data) <= MAX_PER_CATEGORY:

        selected = category_data


    else:

        # Reproducible random sample.

        selected = category_data.sample(
            n=MAX_PER_CATEGORY,
            random_state=42
        )


    balanced_parts.append(
        selected
    )


    print(
        f"{canonical_id} - "
        f"{category_name}: "
        f"{len(category_data):,} -> "
        f"{len(selected):,}"
    )


if not balanced_parts:

    raise ValueError(
        "No records remained after balancing."
    )


training = pd.concat(
    balanced_parts,
    ignore_index=True
)


print()
print(
    f"Records before balancing: "
    f"{before_balancing:,}"
)

print(
    f"Records after balancing: "
    f"{len(training):,}"
)


# ============================================================
# FINAL COLUMNS
# ============================================================

final_columns = [
    "vch_desc",
    "vch_type",
    "canonical_id",
    "canonical_family",
    "label_source",
    "label_confidence",
    "validation_status",
]


training = training[
    final_columns
].copy()


# ============================================================
# SORT
# ============================================================

training = training.sort_values(
    by=[
        "canonical_id",
        "canonical_family",
        "vch_desc",
    ],
    na_position="last"
).reset_index(
    drop=True
)


# ============================================================
# SAVE TRAINING DATASET
# ============================================================

training.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# BUILD LABEL DISTRIBUTION
# ============================================================

label_distribution = (
    training
    .groupby(
        [
            "canonical_id",
            "canonical_family",
        ],
        dropna=False
    )
    .size()
    .reset_index(
        name="record_count"
    )
)


# ============================================================
# BUILD SOURCE DISTRIBUTION
# ============================================================

source_distribution = (
    training
    .groupby(
        [
            "label_source"
        ],
        dropna=False
    )
    .size()
    .reset_index(
        name="record_count"
    )
)


# ============================================================
# BUILD SUMMARY
# ============================================================

summary_rows = []


summary_rows.append({
    "metric":
        "source_transactions",
    "value":
        len(df),
})


summary_rows.append({
    "metric":
        "approved_mapping_records",
    "value":
        len(approved),
})


summary_rows.append({
    "metric":
        "rule_mapping_records",
    "value":
        len(rule_labels),
})


summary_rows.append({
    "metric":
        "ml_generated_records_excluded",
    "value":
        len(ml_labels),
})


summary_rows.append({
    "metric":
        "strong_shopping_labels",
    "value":
        len(shopping_training),
})


summary_rows.append({
    "metric":
        "conflicting_descriptions_removed",
    "value":
        len(conflicting_descriptions),
})


summary_rows.append({
    "metric":
        "final_training_records",
    "value":
        len(training),
})


summary_rows.append({
    "metric":
        "max_records_per_category",
    "value":
        MAX_PER_CATEGORY,
})


# ============================================================
# ADD LABEL DISTRIBUTION TO SUMMARY
# ============================================================

for _, row in label_distribution.iterrows():

    summary_rows.append({

        "metric":
            (
                f"label_"
                f"{int(row['canonical_id'])}_"
                f"{row['canonical_family']}"
            ),

        "value":
            int(row["record_count"]),
    })


# ============================================================
# ADD SOURCE DISTRIBUTION TO SUMMARY
# ============================================================

for _, row in source_distribution.iterrows():

    summary_rows.append({

        "metric":
            (
                f"source_"
                f"{row['label_source']}"
            ),

        "value":
            int(row["record_count"]),
    })


summary = pd.DataFrame(
    summary_rows
)


summary.to_csv(
    SUMMARY_FILE,
    index=False
)


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 90)
print("EXPANDED TRUSTED ML TRAINING DATASET CREATED")
print("=" * 90)


print()
print(
    f"Source transactions:                 "
    f"{len(df):,}"
)

print(
    f"Approved mapping records:            "
    f"{len(approved):,}"
)

print(
    f"Deterministic rule records:          "
    f"{len(rule_labels):,}"
)

print(
    f"ML-generated records EXCLUDED:       "
    f"{len(ml_labels):,}"
)

print(
    f"Strong Shopping labels:              "
    f"{len(shopping_training):,}"
)

print(
    f"Conflicting descriptions removed:    "
    f"{len(conflicting_descriptions):,}"
)

print(
    f"Final training records:              "
    f"{len(training):,}"
)


print()
print("Final label distribution:")


for _, row in label_distribution.iterrows():

    print(
        f"  {int(row['canonical_id'])} - "
        f"{row['canonical_family']}: "
        f"{int(row['record_count']):,}"
    )


print()
print("Training label sources:")


for _, row in source_distribution.iterrows():

    print(
        f"  {row['label_source']}: "
        f"{int(row['record_count']):,}"
    )


print()
print("Output files:")
print(OUTPUT_FILE)
print(SUMMARY_FILE)


