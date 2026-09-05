from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "hk_transactions_with_canonical_v2.csv"

print("=" * 80)
print("AUDITING DETERMINISTIC RULE MAPPINGS")
print("=" * 80)

df = pd.read_csv(DATA_PATH, low_memory=False)

print(f"\nTotal transactions: {len(df):,}")

# ---------------------------------------------------------
# FILTER RULE-MAPPED TRANSACTIONS
# ---------------------------------------------------------

rule_df = df[
    df["mapping_method"].astype(str).str.upper().eq("RULE")
].copy()

print(f"RULE-mapped transactions: {len(rule_df):,}")

if rule_df.empty:
    print("\nNo RULE mappings found.")
    raise SystemExit

# ---------------------------------------------------------
# NORMALIZE DESCRIPTION
# ---------------------------------------------------------

rule_df["description_clean"] = (
    rule_df["vch_desc"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.lower()
)

# ---------------------------------------------------------
# TOP EXACT DESCRIPTION MAPPINGS
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("TOP 100 RULE-MAPPED DESCRIPTIONS")
print("=" * 80)

top_descriptions = (
    rule_df
    .groupby(["description_clean", "canonical_family"])
    .size()
    .reset_index(name="transaction_count")
    .sort_values("transaction_count", ascending=False)
    .head(100)
)

print(
    top_descriptions[
        ["description_clean", "canonical_family", "transaction_count"]
    ].to_string(index=False)
)

# ---------------------------------------------------------
# FAMILY TOTALS
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("RULE MAPPINGS BY CANONICAL FAMILY")
print("=" * 80)

family_summary = (
    rule_df["canonical_family"]
    .value_counts()
    .rename_axis("canonical_family")
    .reset_index(name="transaction_count")
)

print(family_summary.to_string(index=False))

# ---------------------------------------------------------
# UNIQUE DESCRIPTIONS BY FAMILY
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("UNIQUE DESCRIPTIONS BY FAMILY")
print("=" * 80)

unique_family = (
    rule_df
    .groupby("canonical_family")["description_clean"]
    .nunique()
    .reset_index(name="unique_descriptions")
    .sort_values("unique_descriptions", ascending=False)
)

print(unique_family.to_string(index=False))

# ---------------------------------------------------------
# POTENTIAL GENERIC / QUESTIONABLE DESCRIPTIONS
# ---------------------------------------------------------

generic_terms = [
    "online",
    "international",
    "bike",
    "parking",
    "car wash",
    "haircut",
    "hair cut",
    "cutting",
    "adjustment",
    "balance",
    "misc",
    "ammi",
    "ghar",
    "payment",
    "cash",
    "card",
    "store",
    "mart",
    "shopping",
    "ride",
    "transport",
    "gas",
    "oil",
]

print("\n" + "=" * 80)
print("POTENTIALLY QUESTIONABLE / GENERIC RULE MAPPINGS")
print("=" * 80)

generic_pattern = "|".join(
    term.replace(" ", r"\s+")
    for term in generic_terms
)

questionable = rule_df[
    rule_df["description_clean"].str.contains(
        generic_pattern,
        case=False,
        regex=True,
        na=False
    )
]

questionable_summary = (
    questionable
    .groupby(["description_clean", "canonical_family"])
    .size()
    .reset_index(name="transaction_count")
    .sort_values("transaction_count", ascending=False)
    .head(100)
)

if questionable_summary.empty:
    print("No potentially questionable mappings found.")
else:
    print(questionable_summary.to_string(index=False))

# ---------------------------------------------------------
# RULE CONFLICTS
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("DESCRIPTION → MULTIPLE CANONICAL FAMILIES")
print("=" * 80)

conflicts = (
    rule_df
    .groupby("description_clean")["canonical_family"]
    .nunique()
    .reset_index(name="family_count")
)

conflicting_descriptions = conflicts[
    conflicts["family_count"] > 1
].sort_values("family_count", ascending=False)

if conflicting_descriptions.empty:
    print("No description conflicts found.")
else:
    conflict_details = (
        rule_df[
            rule_df["description_clean"].isin(
                conflicting_descriptions["description_clean"]
            )
        ]
        .groupby(
            ["description_clean", "canonical_family"]
        )
        .size()
        .reset_index(name="transaction_count")
        .sort_values(
            ["description_clean", "transaction_count"],
            ascending=[True, False]
        )
    )

    print(conflict_details.to_string(index=False))

# ---------------------------------------------------------
# RULE CONTRIBUTION
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("RULE CONTRIBUTION")
print("=" * 80)

total_auto = df[
    df["mapping_method"].astype(str).str.upper().ne("UNMAPPED")
].shape[0]

print(f"RULE mappings: {len(rule_df):,}")
print(f"All automatic mappings: {total_auto:,}")

if total_auto > 0:
    print(
        f"RULE share of automatic mappings: "
        f"{len(rule_df) / total_auto * 100:.2f}%"
    )
