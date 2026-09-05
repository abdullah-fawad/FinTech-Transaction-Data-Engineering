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
print("ANALYZING ORIGINAL CATEGORY_ID → CANONICAL CATEGORY")
print("=" * 90)

print(f"\nLoading:")
print(INPUT_PATH)

df = pd.read_csv(INPUT_PATH, low_memory=False)

print(f"Transactions loaded: {len(df):,}")


# ============================================================
# BASIC CLEANING
# ============================================================

df["canonical_family_clean"] = (
    df["canonical_family"]
    .fillna("UNMAPPED")
    .astype(str)
    .str.strip()
)

df["category_id_clean"] = (
    df["category_id"]
    .fillna(-1)
)


# ============================================================
# ALL CATEGORY IDs
# ============================================================

print("\n" + "=" * 90)
print("CATEGORY ID OVERVIEW")
print("=" * 90)

print(
    f"Unique category IDs: "
    f"{df['category_id_clean'].nunique():,}"
)


# ============================================================
# ONLY ALREADY-MAPPED TRANSACTIONS
# ============================================================

mapped = df[
    (df["canonical_family_clean"] != "UNMAPPED")
    & (df["mapping_method"].isin([
        "APPROVED_MAPPING",
        "RULE"
    ]))
]

print("\n" + "=" * 90)
print("MAPPED TRANSACTION DATA")
print("=" * 90)

print(f"Mapped transactions: {len(mapped):,}")


# ============================================================
# CATEGORY ID → CANONICAL FAMILY
# ============================================================

print("\n" + "=" * 90)
print("CATEGORY_ID → CANONICAL FAMILY ANALYSIS")
print("=" * 90)

mapping_counts = (
    mapped
    .groupby(
        ["category_id_clean", "canonical_family_clean"]
    )
    .size()
    .reset_index(name="transaction_count")
)

# Total transactions for each category ID
category_totals = (
    mapped
    .groupby("category_id_clean")
    .size()
    .reset_index(name="mapped_total")
)

analysis = mapping_counts.merge(
    category_totals,
    on="category_id_clean",
    how="left"
)

analysis["percentage"] = (
    analysis["transaction_count"]
    / analysis["mapped_total"]
    * 100
)

analysis = analysis.sort_values(
    ["category_id_clean", "transaction_count"],
    ascending=[True, False]
)


# ============================================================
# PRINT CATEGORY ID DISTRIBUTION
# ============================================================

for category_id in sorted(
    analysis["category_id_clean"].unique()
):

    subset = analysis[
        analysis["category_id_clean"] == category_id
    ]

    print(
        f"\nCATEGORY_ID: {category_id}"
    )

    for _, row in subset.iterrows():

        print(
            f"    "
            f"{row['canonical_family_clean']:<30} "
            f"{int(row['transaction_count']):>8,} "
            f"({row['percentage']:>6.2f}%)"
        )


# ============================================================
# FIND DOMINANT CATEGORY
# ============================================================

dominant = (
    analysis
    .sort_values(
        ["category_id_clean", "transaction_count"],
        ascending=[True, False]
    )
    .groupby("category_id_clean")
    .first()
    .reset_index()
)

dominant = dominant[
    [
        "category_id_clean",
        "canonical_family_clean",
        "transaction_count",
        "mapped_total",
        "percentage"
    ]
].copy()

dominant = dominant.rename(
    columns={
        "canonical_family_clean": "dominant_canonical_family",
        "transaction_count": "dominant_count",
        "percentage": "dominant_percentage"
    }
)


# ============================================================
# CLASSIFY CATEGORY IDs
# ============================================================

def classify_row(row):

    if row["dominant_percentage"] >= 95:
        return "STRONG"

    elif row["dominant_percentage"] >= 80:
        return "MODERATE"

    elif row["dominant_percentage"] >= 60:
        return "WEAK"

    else:
        return "AMBIGUOUS"


dominant["mapping_strength"] = dominant.apply(
    classify_row,
    axis=1
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 90)
print("CATEGORY_ID MAPPING STRENGTH SUMMARY")
print("=" * 90)

strength_counts = (
    dominant["mapping_strength"]
    .value_counts()
)

for strength, count in strength_counts.items():

    print(
        f"{strength:<15} "
        f"{count:>6,} category IDs"
    )


# ============================================================
# STRONG CATEGORY IDs
# ============================================================

print("\n" + "=" * 90)
print("STRONG CATEGORY_ID MAPPINGS (>=95%)")
print("=" * 90)

strong = dominant[
    dominant["mapping_strength"] == "STRONG"
].sort_values(
    "category_id_clean"
)

for _, row in strong.iterrows():

    print(
        f"category_id={row['category_id_clean']} "
        f"→ {row['dominant_canonical_family']} "
        f"| {int(row['dominant_count']):,} / "
        f"{int(row['mapped_total']):,} "
        f"({row['dominant_percentage']:.2f}%)"
    )


# ============================================================
# MODERATE CATEGORY IDs
# ============================================================

print("\n" + "=" * 90)
print("MODERATE CATEGORY_ID MAPPINGS (80%-94.99%)")
print("=" * 90)

moderate = dominant[
    dominant["mapping_strength"] == "MODERATE"
].sort_values(
    "category_id_clean"
)

for _, row in moderate.iterrows():

    print(
        f"category_id={row['category_id_clean']} "
        f"→ {row['dominant_canonical_family']} "
        f"| {int(row['dominant_count']):,} / "
        f"{int(row['mapped_total']):,} "
        f"({row['dominant_percentage']:.2f}%)"
    )


# ============================================================
# AMBIGUOUS CATEGORY IDs
# ============================================================

print("\n" + "=" * 90)
print("AMBIGUOUS CATEGORY IDs")
print("=" * 90)

ambiguous_ids = dominant[
    dominant["mapping_strength"].isin(
        ["WEAK", "AMBIGUOUS"]
    )
]["category_id_clean"].tolist()

if len(ambiguous_ids) == 0:

    print("No ambiguous category IDs found.")

else:

    print(
        f"Ambiguous category IDs: "
        f"{len(ambiguous_ids):,}"
    )

    for category_id in ambiguous_ids:

        subset = analysis[
            analysis["category_id_clean"] == category_id
        ]

        print(
            f"\ncategory_id={category_id}"
        )

        for _, row in subset.iterrows():

            print(
                f"    "
                f"{row['canonical_family_clean']:<30} "
                f"{int(row['transaction_count']):>8,} "
                f"({row['percentage']:>6.2f}%)"
            )


# ============================================================
# HOW MANY UNRESOLVED COULD BE MAPPED?
# ============================================================

strong_mapping = dominant[
    dominant["mapping_strength"] == "STRONG"
][
    [
        "category_id_clean",
        "dominant_canonical_family",
        "dominant_percentage"
    ]
].copy()

unresolved = df[
    df["canonical_family_clean"] == "UNMAPPED"
].copy()

unresolved_test = unresolved.merge(
    strong_mapping,
    on="category_id_clean",
    how="left"
)

potential_mask = (
    unresolved_test["dominant_canonical_family"].notna()
)

potential_count = potential_mask.sum()

print("\n" + "=" * 90)
print("POTENTIAL COVERAGE FROM STRONG CATEGORY_ID MAPPINGS")
print("=" * 90)

print(
    f"Unresolved transactions: "
    f"{len(unresolved):,}"
)

print(
    f"Could potentially be mapped using "
    f"STRONG category_id mappings: "
    f"{potential_count:,}"
)

print(
    f"Potential additional coverage: "
    f"{potential_count / len(df) * 100:.2f}% "
    f"of all transactions"
)


# ============================================================
# BREAKDOWN OF POTENTIAL MAPPINGS
# ============================================================

print("\n" + "=" * 90)
print("POTENTIAL ADDITIONAL MAPPINGS BY CANONICAL FAMILY")
print("=" * 90)

potential = unresolved_test[
    potential_mask
]

potential_breakdown = (
    potential["dominant_canonical_family"]
    .value_counts()
)

for family, count in potential_breakdown.items():

    print(
        f"{family:<30} "
        f"{count:>8,}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

analysis_path = (
    OUTPUT_DIR /
    "category_id_canonical_analysis.csv"
)

dominant_path = (
    OUTPUT_DIR /
    "category_id_dominant_mapping.csv"
)

analysis.to_csv(
    analysis_path,
    index=False
)

dominant.to_csv(
    dominant_path,
    index=False
)
