from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = ROOT / "data" / "hk_transactions_with_canonical_v2.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 90)
print("ANALYZING UNRESOLVED TRANSACTION METADATA")
print("=" * 90)

print(f"\nLoading:")
print(INPUT_PATH)

df = pd.read_csv(INPUT_PATH, low_memory=False)

print(f"Transactions loaded: {len(df):,}")


# ============================================================
# FILTER UNRESOLVED
# ============================================================

unresolved = df[
    df["canonical_family"].isna()
    | (df["canonical_family"].astype(str).str.strip() == "")
    | (df["canonical_family"].astype(str).str.strip() == "UNMAPPED")
].copy()

print(f"\nUnresolved transactions: {len(unresolved):,}")


# ============================================================
# USE CASE TITLE
# ============================================================

print("\n" + "=" * 90)
print("TOP USE_CASE_TITLE VALUES")
print("=" * 90)

use_case = (
    unresolved["use_case_title"]
    .fillna("")
    .astype(str)
    .str.strip()
)

use_case = use_case[
    ~use_case.str.lower().isin(["", "nan", "none", "null"])
]

print(f"Non-blank use_case_title values: {len(use_case):,}")
print(f"Unique use_case_title values: {use_case.nunique():,}")

print("\nTop 100:")

for value, count in use_case.value_counts().head(100).items():
    print(f"{count:>8,}  {value}")


# ============================================================
# ORIGINAL CATEGORY ID
# ============================================================

print("\n" + "=" * 90)
print("TOP ORIGINAL CATEGORY_ID VALUES")
print("=" * 90)

category = unresolved["category_id"].copy()

print(
    f"Unique category_id values: "
    f"{category.nunique(dropna=True):,}"
)

for value, count in category.value_counts(dropna=False).head(100).items():
    print(f"{count:>8,}  {value}")


# ============================================================
# CATEGORY ID + USE CASE TITLE
# ============================================================

print("\n" + "=" * 90)
print("CATEGORY_ID + USE_CASE_TITLE COMBINATIONS")
print("=" * 90)

combo = (
    unresolved[
        ["category_id", "use_case_title"]
    ]
    .copy()
)

combo["use_case_title"] = (
    combo["use_case_title"]
    .fillna("")
    .astype(str)
    .str.strip()
)

combo["use_case_title"] = combo["use_case_title"].replace(
    {
        "": "MISSING",
        "nan": "MISSING",
        "None": "MISSING",
        "null": "MISSING"
    }
)

combo_counts = (
    combo
    .value_counts()
    .head(100)
)

for (category_id, title), count in combo_counts.items():
    print(
        f"{count:>8,}  "
        f"category_id={category_id} | "
        f"use_case_title={title}"
    )


# ============================================================
# DESCRIPTION + CATEGORY + USE CASE
# ============================================================

print("\n" + "=" * 90)
print("SAMPLE UNRESOLVED TRANSACTIONS")
print("=" * 90)

sample_columns = [
    "voucher_id",
    "vch_type",
    "vch_amount",
    "vch_desc",
    "category_id",
    "use_case_title"
]

sample_columns = [
    col for col in sample_columns
    if col in unresolved.columns
]

print(
    unresolved[
        sample_columns
    ]
    .head(100)
    .to_string(index=False)
)


# ============================================================
# SAVE METADATA ANALYSIS
# ============================================================

OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

output_path = OUTPUT_DIR / "unresolved_metadata.csv"

unresolved.to_csv(
    output_path,
    index=False
)

print("\n" + "=" * 90)
print("ANALYSIS COMPLETE")
print("=" * 90)

print(f"\nDetailed output saved to:")
print(output_path)