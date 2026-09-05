from pathlib import Path
import pandas as pd
import re

# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = ROOT / "data" / "hk_transactions_with_canonical_v2.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 90)
print("DIAGNOSTIC ANALYSIS OF UNRESOLVED TRANSACTIONS")
print("=" * 90)

print(f"\nLoading:")
print(INPUT_PATH)

df = pd.read_csv(INPUT_PATH, low_memory=False)

print(f"Transactions loaded: {len(df):,}")


# ============================================================
# BASIC VALIDATION
# ============================================================

required_columns = [
    "vch_desc",
    "vch_type",
    "canonical_family",
    "mapping_method"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# IDENTIFY UNRESOLVED TRANSACTIONS
# ============================================================

unresolved = df[
    df["canonical_family"].isna()
    | (df["canonical_family"].astype(str).str.strip() == "")
    | (df["canonical_family"].astype(str).str.strip() == "UNMAPPED")
].copy()

print("\n" + "=" * 90)
print("UNRESOLVED TRANSACTIONS")
print("=" * 90)

print(f"Total unresolved: {len(unresolved):,}")
print(
    f"Unresolved percentage: "
    f"{len(unresolved) / len(df) * 100:.2f}%"
)


# ============================================================
# DESCRIPTION CLEANING
# ============================================================

unresolved["description_clean"] = (
    unresolved["vch_desc"]
    .fillna("")
    .astype(str)
    .str.strip()
)

unresolved["description_lower"] = (
    unresolved["description_clean"]
    .str.lower()
)


# ============================================================
# BLANK DESCRIPTIONS
# ============================================================

blank_mask = (
    (unresolved["description_clean"] == "")
    | (
        unresolved["description_clean"]
        .str.lower()
        .isin(["nan", "none", "null"])
    )
)

blank_count = blank_mask.sum()
non_blank_count = len(unresolved) - blank_count

print("\n" + "=" * 90)
print("DESCRIPTION QUALITY")
print("=" * 90)

print(f"Blank descriptions:     {blank_count:,}")
print(
    f"Blank percentage:        "
    f"{blank_count / len(unresolved) * 100:.2f}%"
)

print(f"Non-blank descriptions:  {non_blank_count:,}")
print(
    f"Non-blank percentage:    "
    f"{non_blank_count / len(unresolved) * 100:.2f}%"
)


# ============================================================
# DESCRIPTION LENGTH
# ============================================================

unresolved["description_length"] = (
    unresolved["description_clean"].str.len()
)

print("\n" + "=" * 90)
print("DESCRIPTION LENGTH")
print("=" * 90)

length_bands = {
    "0 characters": unresolved["description_length"] == 0,
    "1-5 characters": unresolved["description_length"].between(1, 5),
    "6-10 characters": unresolved["description_length"].between(6, 10),
    "11-20 characters": unresolved["description_length"].between(11, 20),
    "21-50 characters": unresolved["description_length"].between(21, 50),
    "51+ characters": unresolved["description_length"] >= 51,
}

for label, mask in length_bands.items():
    count = mask.sum()
    percentage = count / len(unresolved) * 100

    print(
        f"{label:<20} "
        f"{count:>8,} "
        f"({percentage:>6.2f}%)"
    )


# ============================================================
# MOST COMMON UNRESOLVED DESCRIPTIONS
# ============================================================

print("\n" + "=" * 90)
print("TOP 50 MOST COMMON UNRESOLVED DESCRIPTIONS")
print("=" * 90)

common_descriptions = (
    unresolved.loc[
        unresolved["description_clean"] != "",
        "description_lower"
    ]
    .value_counts()
    .head(50)
)

if len(common_descriptions) == 0:
    print("No non-blank descriptions found.")

else:
    for description, count in common_descriptions.items():
        print(f"{count:>8,}  {description}")


# ============================================================
# TRANSACTION TYPE
# ============================================================

print("\n" + "=" * 90)
print("UNRESOLVED TRANSACTIONS BY VCH_TYPE")
print("=" * 90)

type_counts = (
    unresolved["vch_type"]
    .fillna("MISSING")
    .astype(str)
    .str.strip()
    .value_counts()
)

for transaction_type, count in type_counts.items():
    percentage = count / len(unresolved) * 100

    print(
        f"{str(transaction_type):<30} "
        f"{count:>8,} "
        f"({percentage:>6.2f}%)"
    )


# ============================================================
# MAPPING METHOD CHECK
# ============================================================

print("\n" + "=" * 90)
print("MAPPING METHOD FOR UNRESOLVED")
print("=" * 90)

method_counts = (
    unresolved["mapping_method"]
    .fillna("MISSING")
    .astype(str)
    .value_counts()
)

print(method_counts)


# ============================================================
# CHECK OTHER COLUMNS
# ============================================================

print("\n" + "=" * 90)
print("AVAILABLE COLUMNS")
print("=" * 90)

for column in df.columns:
    print(f"- {column}")


# ============================================================
# NON-NULL COUNTS FOR POTENTIALLY USEFUL FIELDS
# ============================================================

potential_fields = [
    "vch_desc",
    "vch_type",
    "category_id",
    "source_category_id",
    "use_case_title",
    "source_id",
    "consumer_id",
    "amount",
    "foreign_currency_amount",
    "foreign_currency_rate",
    "account_id"
]

print("\n" + "=" * 90)
print("DATA AVAILABILITY IN UNRESOLVED TRANSACTIONS")
print("=" * 90)

for column in potential_fields:

    if column not in unresolved.columns:
        continue

    non_null = unresolved[column].notna().sum()
    percentage = non_null / len(unresolved) * 100

    print(
        f"{column:<30} "
        f"{non_null:>8,} "
        f"({percentage:>6.2f}% non-null)"
    )


# ============================================================
# UNIQUE DESCRIPTION ANALYSIS
# ============================================================

non_blank = unresolved[
    unresolved["description_clean"] != ""
].copy()

print("\n" + "=" * 90)
print("UNIQUE DESCRIPTION ANALYSIS")
print("=" * 90)

if len(non_blank) > 0:

    unique_descriptions = (
        non_blank["description_lower"]
        .nunique()
    )

    print(
        f"Unique non-blank descriptions: "
        f"{unique_descriptions:,}"
    )

    print(
        f"Total non-blank transactions: "
        f"{len(non_blank):,}"
    )

    print(
        f"Transactions per unique description: "
        f"{len(non_blank) / unique_descriptions:.2f}"
    )

else:
    print("No non-blank descriptions.")


# ============================================================
# POTENTIALLY USEFUL KEYWORDS
# ============================================================

keyword_groups = {
    "Food": [
        "food",
        "restaurant",
        "grocery",
        "groceries",
        "mart",
        "bakery",
        "cafe",
        "pizza",
        "kfc",
        "mcdonald",
        "foodpanda"
    ],

    "Fuel": [
        "petrol",
        "fuel",
        "shell",
        "pso",
        "parco",
        "hascol",
        "attock",
        "caltex"
    ],

    "Transport": [
        "uber",
        "careem",
        "indrive",
        "yango",
        "bykea",
        "taxi",
        "ride"
    ],

    "Utilities": [
        "electricity",
        "electric",
        "gas bill",
        "water bill",
        "utility",
        "k-electric",
        "ssgc",
        "sngpl",
        "ptcl",
        "internet"
    ],

    "Healthcare": [
        "hospital",
        "clinic",
        "pharmacy",
        "medical",
        "medicine",
        "doctor",
        "laboratory"
    ],

    "Shopping": [
        "daraz",
        "amazon",
        "shopping",
        "clothing",
        "clothes",
        "mall",
        "retail",
        "store"
    ],

    "Income": [
        "salary",
        "wages",
        "payroll",
        "income"
    ],

    "Transfers": [
        "transfer",
        "withdrawal",
        "atm",
        "cash withdrawal"
    ]
}

print("\n" + "=" * 90)
print("POTENTIALLY CLASSIFIABLE BY KEYWORD")
print("=" * 90)

keyword_match_counts = {}

for group, keywords in keyword_groups.items():

    pattern = "|".join(
        [re.escape(k) for k in keywords]
    )

    mask = (
        unresolved["description_lower"]
        .str.contains(
            pattern,
            regex=True,
            na=False
        )
    )

    count = mask.sum()

    keyword_match_counts[group] = count

    print(
        f"{group:<20} "
        f"{count:>8,} "
        f"({count / len(unresolved) * 100:>6.2f}%)"
    )


# ============================================================
# SAVE DIAGNOSTIC OUTPUT
# ============================================================

OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

output_path = OUTPUT_DIR / "unresolved_diagnostic.csv"

unresolved.to_csv(
    output_path,
    index=False
)

print("\n" + "=" * 90)
print("DIAGNOSTIC COMPLETE")
print("=" * 90)

print(f"\nDetailed unresolved records saved to:")
print(output_path)