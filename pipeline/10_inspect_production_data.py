import os
from pathlib import Path

import pandas as pd


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "hk_transactions_with_canonical_v2.csv"
)


# =============================================================================
# CONFIGURATION
# =============================================================================

REQUIRED_COLUMNS = [
    "voucher_id",
    "consumer_id",
    "vch_type",
    "vch_amount",
    "vch_date",
    "vch_desc",
    "category_id",
    "canonical_id",
    "canonical_family",
    "canonical_mapping_status",
]


# =============================================================================
# HEADER
# =============================================================================

print("=" * 90)
print("INSPECTING PRODUCTION DATA")
print("=" * 90)
print()

print("Loading:")
print(INPUT_FILE)
print()


# =============================================================================
# LOAD DATA
# =============================================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Production input file not found: {INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print(f"Transactions loaded: {len(df):,}")
print()


# =============================================================================
# COLUMN VALIDATION
# =============================================================================

print("=" * 90)
print("COLUMN VALIDATION")
print("=" * 90)

missing_columns = [
    column
    for column in REQUIRED_COLUMNS
    if column not in df.columns
]

if missing_columns:
    print("Missing required columns:")
    for column in missing_columns:
        print(f"  - {column}")

    raise ValueError(
        f"Required columns are missing: {missing_columns}"
    )

print("All required columns are present.")
print()


# =============================================================================
# BASIC DATA INSPECTION
# =============================================================================

print("=" * 90)
print("DATASET SHAPE")
print("=" * 90)

print(f"Rows:    {df.shape[0]:,}")
print(f"Columns: {df.shape[1]:,}")
print()


# =============================================================================
# DUPLICATE CHECK
# =============================================================================

print("=" * 90)
print("DUPLICATE CHECK")
print("=" * 90)

duplicate_vouchers = df["voucher_id"].duplicated().sum()

print(f"Duplicate voucher IDs: {duplicate_vouchers:,}")
print()


# =============================================================================
# NULL CHECK
# =============================================================================

print("=" * 90)
print("NULL VALUE CHECK")
print("=" * 90)

for column in REQUIRED_COLUMNS:
    null_count = df[column].isna().sum()
    print(f"{column:30s}: {null_count:,}")

print()


# =============================================================================
# TRANSACTION TYPE CHECK
# =============================================================================

print("=" * 90)
print("TRANSACTION TYPE DISTRIBUTION")
print("=" * 90)

print(df["vch_type"].value_counts(dropna=False))
print()


# =============================================================================
# CANONICAL MAPPING STATUS
# =============================================================================

print("=" * 90)
print("CANONICAL MAPPING STATUS")
print("=" * 90)

print(
    df["canonical_mapping_status"]
    .value_counts(dropna=False)
)

print()


# =============================================================================
# CANONICAL CATEGORY DISTRIBUTION
# =============================================================================

print("=" * 90)
print("CANONICAL CATEGORY DISTRIBUTION")
print("=" * 90)

canonical_distribution = (
    df.groupby(
        ["canonical_id", "canonical_family"],
        dropna=False
    )
    .size()
    .sort_values(ascending=False)
)

print(canonical_distribution)
print()


# =============================================================================
# ORIGINAL CATEGORY DISTRIBUTION
# =============================================================================

print("=" * 90)
print("ORIGINAL CATEGORY ID DISTRIBUTION")
print("=" * 90)

print(
    df["category_id"]
    .value_counts(dropna=False)
    .sort_index()
)

print()


# =============================================================================
# AMOUNT CHECK
# =============================================================================

print("=" * 90)
print("AMOUNT CHECK")
print("=" * 90)

print(f"Minimum amount: {df['vch_amount'].min()}")
print(f"Maximum amount: {df['vch_amount'].max()}")
print(f"Negative amounts: {(df['vch_amount'] < 0).sum():,}")
print(f"Zero amounts: {(df['vch_amount'] == 0).sum():,}")
print()


# =============================================================================
# DATE CHECK
# =============================================================================

print("=" * 90)
print("DATE CHECK")
print("=" * 90)

dates = pd.to_datetime(
    df["vch_date"],
    errors="coerce"
)

invalid_dates = dates.isna().sum()

print(f"Invalid/missing transaction dates: {invalid_dates:,}")

if dates.notna().any():
    print(f"Earliest transaction date: {dates.min()}")
    print(f"Latest transaction date:   {dates.max()}")

    future_dates = (dates > pd.Timestamp.now()).sum()
    print(f"Future transaction dates:   {future_dates:,}")

print()


# =============================================================================
# DESCRIPTION CHECK
# =============================================================================

print("=" * 90)
print("DESCRIPTION CHECK")
print("=" * 90)

missing_descriptions = (
    df["vch_desc"]
    .isna()
    .sum()
)

empty_descriptions = (
    df["vch_desc"]
    .fillna("")
    .astype(str)
    .str.strip()
    .eq("")
    .sum()
)

print(f"Missing descriptions: {missing_descriptions:,}")
print(f"Empty descriptions:   {empty_descriptions:,}")
print()


# =============================================================================
# DATA QUALITY SUMMARY
# =============================================================================

print("=" * 90)
print("DATA QUALITY SUMMARY")
print("=" * 90)

quality_checks = {
    "Total transactions": len(df),
    "Duplicate voucher IDs": duplicate_vouchers,
    "Missing voucher IDs": df["voucher_id"].isna().sum(),
    "Missing consumer IDs": df["consumer_id"].isna().sum(),
    "Missing descriptions": missing_descriptions,
    "Missing transaction dates": df["vch_date"].isna().sum(),
    "Negative amounts": (df["vch_amount"] < 0).sum(),
    "Zero amounts": (df["vch_amount"] == 0).sum(),
    "Missing canonical IDs": df["canonical_id"].isna().sum(),
    "Missing canonical families": df["canonical_family"].isna().sum(),
}

for check, value in quality_checks.items():
    print(f"{check:35s}: {value:,}")

print()


# =============================================================================
# SAMPLE RECORDS
# =============================================================================

print("=" * 90)
print("SAMPLE PRODUCTION RECORDS")
print("=" * 90)

sample_columns = [
    "voucher_id",
    "consumer_id",
    "vch_type",
    "vch_amount",
    "vch_date",
    "vch_desc",
    "category_id",
    "canonical_id",
    "canonical_family",
    "canonical_mapping_status",
]

print(
    df[sample_columns]
    .head(10)
    .to_string(index=False)
)

print()


# =============================================================================
# FINAL STATUS
# =============================================================================

print("=" * 90)
print("PRODUCTION DATA INSPECTION COMPLETE")
print("=" * 90)

print()
print(f"Source file:       {INPUT_FILE}")
print(f"Transactions:      {len(df):,}")
print(f"Columns:           {len(df.columns):,}")
print(f"Duplicate IDs:     {duplicate_vouchers:,}")
print(f"Missing desc.:     {missing_descriptions:,}")
print(f"Missing dates:     {df['vch_date'].isna().sum():,}")
print()
print("STATUS: PASS")