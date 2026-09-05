from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.normalize import normalize


ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "raw" / "hk_transactions_table.csv"
OUT_DIR = ROOT / "outputs"


MAX_SANE_CATEGORY_ID = 10_000


# ---------------------------------------------------------------------
# CANONICAL BUSINESS FAMILIES
# ---------------------------------------------------------------------

FAMILY_PATTERNS = {
    "Fuel": [
        "petrol",
        "fuel",
        "diesel",
        "gasoline",
        "cng",
        "benzine",
        "petroleum",
    ],

    "Transfers": [
        "cash",
        "atm",
        "transfer",
        "withdrawal",
        "withdrawl",
        "cash withdrawal",
        "cash withdraw",
        "cash withdrawl",
    ],

    "Utilities": [
        "electricity",
        "utility",
        "water bill",
        "gas bill",
        "ptcl",
        "cable",
        "electricity bill",
    ],

    "Dining": [
        "lunch",
        "dinner",
        "breakfast",
        "roti",
        "biryani",
        "pizza",
        "burger",
        "kfc",
        "mcdonald",
        "restaurant",
        "food",
    ],

    "Groceries": [
        "grocery",
        "groceries",
        "milk",
        "doodh",
        "bread",
        "eggs",
        "egg",
        "fruit",
        "fruits",
        "vegetables",
        "vegetable",
        "sabzi",
        "kirana",
        "lauka",
        "karela",
        "bajra",
        "dahi",
    ],

    "Healthcare": [
        "medicine",
        "medicines",
        "doctor",
        "hospital",
        "medical",
        "clinic",
        "pharmacy",
    ],

    "Shopping": [
        "shoes",
        "clothes",
        "shopping",
        "clothing",
        "flipkart",
        "daraz",
        "store",
        "mall",
    ],

    "Income": [
        "salary",
        "interest earned",
        "online deposit",
        "income",
        "deposit",
    ],

    "Transport": [
        "careem",
        "uber",
        "parking",
        "taxi",
        "toll",
        "toll tax",
        "rickshaw",
    ],
}


CANONICAL_IDS = {
    "Dining": 2001,
    "Fuel": 2002,
    "Transport": 2003,
    "Utilities": 2004,
    "Healthcare": 2005,
    "Shopping": 2006,
    "Transfers": 2007,
    "Income": 2008,
    "Groceries": 2009,
    "Other": 2010,
}


# ---------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------

def contains_term(description: str, term: str) -> bool:
    """Check whether a term occurs in a normalized description."""

    description = str(description).lower().strip()
    term = str(term).lower().strip()

    if not description or not term:
        return False

    # Multi-word phrases can be searched directly.
    if " " in term:
        return term in description

    # Single words should be matched as complete tokens.
    return term in description.split()


def proposed_family(description: str, vch_type: str) -> str:
    """
    Propose a canonical business family.

    Transaction type is used as additional context, but this function
    only creates a proposal. It does NOT approve anything.
    """

    description = str(description).lower().strip()
    vch_type = str(vch_type).strip()

    # -------------------------------------------------------------
    # Strong contextual transfer signals
    # -------------------------------------------------------------

    if vch_type == "Transfer":
        transfer_terms = FAMILY_PATTERNS["Transfers"]

        if any(contains_term(description, term) for term in transfer_terms):
            return "Transfers"

    # -------------------------------------------------------------
    # Strong income signals
    # -------------------------------------------------------------

    if vch_type == "Income":
        income_terms = FAMILY_PATTERNS["Income"]

        if any(contains_term(description, term) for term in income_terms):
            return "Income"

    # -------------------------------------------------------------
    # General family matching
    # -------------------------------------------------------------

    matched_families = []

    for family, terms in FAMILY_PATTERNS.items():

        if family in {"Transfers", "Income"}:
            continue

        if any(contains_term(description, term) for term in terms):
            matched_families.append(family)

    # Exactly one family = clean proposal.
    if len(matched_families) == 1:
        return matched_families[0]

    # Multiple families = ambiguous.
    if len(matched_families) > 1:
        return "AMBIGUOUS"

    return "Unassigned"


def calculate_category_purity(group: pd.DataFrame) -> tuple[float, int]:
    """
    Return dominant source category purity and dominant category ID.
    """

    counts = group["category_id"].value_counts()

    if len(counts) == 0:
        return 0.0, -1

    dominant_category = counts.index[0]
    dominant_count = counts.iloc[0]

    purity = dominant_count / len(group)

    return float(purity), int(dominant_category)


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main() -> None:

    print("=" * 70)
    print("BUILD CANONICAL TAXONOMY REVIEW PROPOSALS")
    print("=" * 70)

    # -------------------------------------------------------------
    # 1. Load original data
    # -------------------------------------------------------------

    df = pd.read_csv(DATA_PATH)

    required_columns = [
        "vch_desc",
        "category_id",
        "vch_type",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df = df.dropna(
        subset=required_columns
    ).copy()

    # -------------------------------------------------------------
    # 2. Keep original category_id untouched
    # -------------------------------------------------------------

    df["source_category_id"] = df["category_id"]

    # Convert only the working copy to numeric.
    df["source_category_id"] = pd.to_numeric(
        df["source_category_id"],
        errors="coerce",
    )

    df = df[
        df["source_category_id"].notna()
    ].copy()

    df = df[
        df["source_category_id"] <= MAX_SANE_CATEGORY_ID
    ].copy()


    # -------------------------------------------------------------
    # 3. Normalize descriptions
    # -------------------------------------------------------------

    df["clean_description"] = (
        df["vch_desc"]
        .astype(str)
        .apply(normalize)
    )

    df = df[
        df["clean_description"].str.strip() != ""
    ].copy()

    df["vch_type"] = (
        df["vch_type"]
        .astype(str)
        .str.strip()
    )

    # -------------------------------------------------------------
    # 4. Generate family proposals
    # -------------------------------------------------------------

    df["proposed_family"] = [
        proposed_family(
            description,
            vch_type,
        )
        for description, vch_type
        in zip(
            df["clean_description"],
            df["vch_type"],
        )
    ]

    # Keep only actual family proposals.
    df = df[
        ~df["proposed_family"].isin(
            ["Unassigned", "AMBIGUOUS"]
        )
    ].copy()

    # -------------------------------------------------------------
    # 5. Group by description + transaction type + family
    # -------------------------------------------------------------

    proposal_groups = []

    grouped = df.groupby(
        [
            "proposed_family",
            "clean_description",
            "vch_type",
        ]
    )

    for (
        family,
        description,
        vch_type,
    ), group in grouped:

        support = len(group)

        purity, dominant_category = (
            calculate_category_purity(group)
        )

        unique_source_categories = (
            group["source_category_id"]
            .nunique()
        )

        canonical_id = CANONICAL_IDS.get(
            family
        )

        proposal_groups.append(
            {
                "description": description,
                "vch_type": vch_type,
                "proposed_family": family,
                "proposed_canonical_id": canonical_id,

                # Evidence
                "transaction_support": support,
                "dominant_source_category_id": dominant_category,
                "distinct_source_category_ids":
                    unique_source_categories,
                "source_category_purity":
                    round(purity, 4),

                # Human decision
                "review_status": "REVIEW_REQUIRED",
                "approved": False,

                "evidence_note": (
                    "Machine-generated proposal based on "
                    "description and transaction type. "
                    "No mapping has been applied."
                ),
            }
        )

    proposals = pd.DataFrame(
        proposal_groups
    )

    if proposals.empty:
        print("\nNo proposals generated.")
        return

    # -------------------------------------------------------------
    # 6. Add risk classification
    # -------------------------------------------------------------

    def classify_risk(row):

        support = row["transaction_support"]
        purity = row["source_category_purity"]

        # Very strong evidence
        if (
            support >= 10
            and purity >= 0.95
        ):
            return "SAFE_CANDIDATE"

        # Reasonably strong but requires review
        if (
            support >= 5
            and purity >= 0.80
        ):
            return "REVIEW_CANDIDATE"

        return "WEAK_CANDIDATE"

    proposals["risk_level"] = proposals.apply(
        classify_risk,
        axis=1,
    )

    # -------------------------------------------------------------
    # 7. Add multilingual / synonym indicator
    # -------------------------------------------------------------

    multilingual_terms = {
        "sabzi": "vegetables",
        "vegetables": "sabzi",
        "doodh": "milk",
        "milk": "doodh",
        "dahi": "yogurt",
        "roti": "bread",
        "lauka": "vegetable",
        "karela": "vegetable",
    }

    proposals["possible_synonym"] = (
        proposals["description"]
        .map(multilingual_terms)
        .fillna("")
    )

    proposals["synonym_review_note"] = ""

    proposals.loc[
        proposals["possible_synonym"] != "",
        "synonym_review_note"
    ] = (
        "Possible multilingual/synonym case. "
        "Review before approval."
    )

    # -------------------------------------------------------------
    # 8. Sort for human review
    # -------------------------------------------------------------

    risk_order = {
        "SAFE_CANDIDATE": 0,
        "REVIEW_CANDIDATE": 1,
        "WEAK_CANDIDATE": 2,
    }

    proposals["_risk_order"] = (
        proposals["risk_level"]
        .map(risk_order)
    )

    proposals = proposals.sort_values(
        [
            "_risk_order",
            "transaction_support",
            "source_category_purity",
        ],
        ascending=[
            True,
            False,
            False,
        ],
    )

    proposals = proposals.drop(
        columns=["_risk_order"]
    )

    # -------------------------------------------------------------
    # 9. Save review file
    # -------------------------------------------------------------

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUT_DIR
        / "canonical_taxonomy_review.csv"
    )

    proposals.to_csv(
        output_path,
        index=False,
    )

    # -------------------------------------------------------------
    # 10. Generate family summary
    # -------------------------------------------------------------

    family_summary = (
        proposals.groupby(
            "proposed_family"
        )
        .agg(
            proposals=(
                "description",
                "count",
            ),
            total_transactions=(
                "transaction_support",
                "sum",
            ),
            distinct_source_categories=(
                "distinct_source_category_ids",
                "sum",
            ),
            median_purity=(
                "source_category_purity",
                "median",
            ),
            safe_candidates=(
                "risk_level",
                lambda x:
                    (x == "SAFE_CANDIDATE").sum(),
            ),
            review_candidates=(
                "risk_level",
                lambda x:
                    (x == "REVIEW_CANDIDATE").sum(),
            ),
            weak_candidates=(
                "risk_level",
                lambda x:
                    (x == "WEAK_CANDIDATE").sum(),
            ),
        )
        .reset_index()
        .sort_values(
            "total_transactions",
            ascending=False,
        )
    )

    family_output = (
        OUT_DIR
        / "canonical_taxonomy_review_summary.csv"
    )

    family_summary.to_csv(
        family_output,
        index=False,
    )

    # -------------------------------------------------------------
    # 11. Print results
    # -------------------------------------------------------------

    print()
    print("=" * 70)
    print("PROPOSAL GENERATION COMPLETE")
    print("=" * 70)

    print(
        f"Unique proposals: "
        f"{len(proposals):,}"
    )

    print(
        f"SAFE candidates: "
        f"{(
            proposals['risk_level']
            == 'SAFE_CANDIDATE'
        ).sum():,}"
    )

    print(
        f"REVIEW candidates: "
        f"{(
            proposals['risk_level']
            == 'REVIEW_CANDIDATE'
        ).sum():,}"
    )

    print(
        f"WEAK candidates: "
        f"{(
            proposals['risk_level']
            == 'WEAK_CANDIDATE'
        ).sum():,}"
    )

    print()
    print("Possible multilingual/synonym cases:")

    synonym_rows = proposals[
        proposals["possible_synonym"] != ""
    ][
        [
            "description",
            "possible_synonym",
            "proposed_family",
            "vch_type",
            "transaction_support",
            "source_category_purity",
        ]
    ]

    if len(synonym_rows) > 0:
        print(
            synonym_rows
            .head(30)
            .to_string(index=False)
        )
    else:
        print("None found.")

    print()
    print(
        f"Review file:\n"
        f"{output_path}"
    )

    print(
        f"\nSummary file:\n"
        f"{family_output}"
    )

if __name__ == "__main__":
    main()