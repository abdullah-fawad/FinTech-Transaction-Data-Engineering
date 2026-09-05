from pathlib import Path
import re
import pandas as pd
import numpy as np
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

RAW_PATH = ROOT / "data" / "raw" / "hk_transactions_table.csv"
APPROVED_MAPPING_PATH = ROOT / "outputs" / "approved_canonical_mappings.csv"
CATEGORY_MAPPING_PATH = ROOT / "outputs" / "category_id_dominant_mapping.csv"
OUTPUT_PATH = ROOT / "data" / "hk_transactions_with_canonical_v2.csv"

MODEL_PATH = (
    ROOT
    / "models"
    / "ml_model"
    / "transaction_category_classifier.joblib"
)


# ============================================================
# ML THRESHOLDS
# ============================================================

ML_CONFIDENCE_THRESHOLD = 0.80
ML_MARGIN_THRESHOLD = 0.20


# ============================================================
# CATEGORY_ID THRESHOLDS
# ============================================================

CATEGORY_ID_MIN_PERCENTAGE = 95.0
CATEGORY_ID_MIN_MAPPED_COUNT = 5


# ============================================================
# CANONICAL TAXONOMY
# ============================================================

CANONICAL_FAMILIES = {
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
    "food": "Food & Groceries",
    "food & groceries": "Food & Groceries",
    "groceries": "Food & Groceries",

    "fuel": "Fuel",

    "transport": "Transport",

    "utilities": "Utilities & Bills",
    "utilities & bills": "Utilities & Bills",
    "utility": "Utilities & Bills",

    "healthcare": "Healthcare",
    "health": "Healthcare",

    "shopping": "Shopping & Retail",
    "shopping & retail": "Shopping & Retail",
    "retail": "Shopping & Retail",

    "transfers": "Transfers & Withdrawals",
    "transfers & withdrawals": "Transfers & Withdrawals",
    "transfer": "Transfers & Withdrawals",

    "income": "Income",

    "other": "Other",
}


def normalize_family(value):

    if pd.isna(value):
        return ""

    text = str(value).strip()

    if not text:
        return ""

    return FAMILY_ALIASES.get(
        text.lower(),
        text
    )


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_description(value):

    if pd.isna(value):
        return ""

    text = str(value).strip().lower()

    if text in {"nan", "none", "null"}:
        return ""

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text


def normalize_type(value):

    if pd.isna(value):
        return ""

    return str(value).strip().lower()


def normalize_category_id(value):

    if pd.isna(value):
        return ""

    text = str(value).strip()

    if text.lower() in {
        "nan",
        "none",
        "null",
        "",
    }:
        return ""

    try:

        numeric = float(text)

        if numeric.is_integer():
            return str(int(numeric))

        return str(numeric)

    except Exception:

        return text

# ============================================================
# DETERMINISTIC DESCRIPTION RULES
# ============================================================

RULE_KEYWORDS = {

    "Fuel": [
        "petrol",
        "gas station",
        "fuel station",
        "petroleum",
        "shell",
        "pso",
        "parco",
        "hascol",
        "attock",
        "caltex",
        "cng",
    ],

    "Transport": [
        "uber",
        "careem",
        "indrive",
        "yango",
        "bykea",
        "taxi",
        "ride",
        "transport",
    ],

    "Utilities & Bills": [
        "electricity",
        "electric bill",
        "gas bill",
        "water bill",
        "utility bill",
        "utilities",
        "k-electric",
        "k electric",
        "ssgc",
        "sngpl",
        "ptcl",
        "internet bill",
        "phone bill",
        "zong internet",
        "jazz internet",
        "telenor internet",
    ],

    "Healthcare": [
        "hospital",
        "clinic",
        "pharmacy",
        "medical",
        "medicine",
        "doctor",
        "laboratory",
        "lab",
        "healthcare",
    ],

    "Shopping & Retail": [
        "daraz",
        "amazon",
        "shopping",
        "clothing",
        "clothes",
        "mall",
        "retail",
        # "store",   # removed: too generic
    ],

    "Food & Groceries": [
        "kfc",
        "mcdonald",
        "pizza",
        "restaurant",
        "food",
        "grocery",
        "groceries",
        # "mart",   # removed: too generic
        "bakery",
        "cafe",
        "foodpanda",
        "cheezious",
        "subway",
        "lunch",
        "dinner",
        "breakfast",
        "nashta",
        "roti",
        "bread",
        "eggs",
        "milk",
        "chicken",
        "fruits",
        "fruit",
        "vegetables",
        "sabzi",
        "biryani",
        "juice",
        "mango",
        "mangoes",
        "tea",
        "chai",
        "dahi",
        "snacks",
        "cake",
    ],

    "Income": [
        "salary",
        "wages",
        "payroll",
        "income",
        "bonus",
    ],

    "Transfers & Withdrawals": [
        "transfer",
        "withdrawal",
        "atm cash",
        "cash withdrawal",
        "funds transfer",
        "credit card payment",
    ],
}

def apply_deterministic_rule(description):

    if not description:
        return None

    matched_families = []

    for family, keywords in RULE_KEYWORDS.items():

        for keyword in keywords:

            if keyword in description:

                matched_families.append(
                    family
                )

                break

    matched_families = list(
        set(matched_families)
    )

    # Only classify when exactly one
    # canonical family matches.

    if len(matched_families) == 1:

        return matched_families[0]

    return None


# ============================================================
# CANONICAL ID HELPER
# ============================================================

def family_to_canonical_id(family):

    for canonical_id, canonical_family in CANONICAL_FAMILIES.items():

        if canonical_family == family:

            return canonical_id

    return np.nan


# ============================================================
# LOAD RAW DATA
# ============================================================

print("=" * 90)
print("LOADING RAW TRANSACTIONS")
print("=" * 90)

df = pd.read_csv(
    RAW_PATH
)

print(
    f"Transactions loaded: "
    f"{len(df):,}"
)


# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

required_raw_columns = [
    "vch_desc",
    "vch_type",
    "category_id",
]

missing_columns = [
    col
    for col in required_raw_columns
    if col not in df.columns
]

if missing_columns:

    raise ValueError(
        f"Missing required raw columns: "
        f"{missing_columns}"
    )


# ============================================================
# PRESERVE ORIGINAL CATEGORY_ID
# ============================================================

df["source_category_id"] = df[
    "category_id"
]


# ============================================================
# NORMALIZE INPUT FIELDS
# ============================================================

df["description_clean"] = df[
    "vch_desc"
].apply(
    normalize_description
)

df["vch_type_clean"] = df[
    "vch_type"
].apply(
    normalize_type
)

df["category_id_clean"] = df[
    "category_id"
].apply(
    normalize_category_id
)


# ============================================================
# INITIALIZE OUTPUT COLUMNS
# ============================================================

df["canonical_id"] = np.nan

df["canonical_family"] = "UNMAPPED"

df["canonical_mapping_status"] = "UNMAPPED"

df["mapping_method"] = "UNMAPPED"

df["ml_confidence"] = np.nan

df["ml_margin"] = np.nan


# ============================================================
# LOAD APPROVED MAPPINGS
# ============================================================

print()
print("=" * 90)
print("LOADING APPROVED DESCRIPTION MAPPINGS")
print("=" * 90)

approved = pd.read_csv(
    APPROVED_MAPPING_PATH
)


required_mapping_columns = [
    "description",
    "vch_type",
    "canonical_id",
    "canonical_family",
]

missing_mapping_columns = [
    col
    for col in required_mapping_columns
    if col not in approved.columns
]

if missing_mapping_columns:

    raise ValueError(
        f"Missing mapping columns: "
        f"{missing_mapping_columns}"
    )


approved["description_clean"] = approved[
    "description"
].apply(
    normalize_description
)

approved["vch_type_clean"] = approved[
    "vch_type"
].apply(
    normalize_type
)

approved["canonical_family"] = approved[
    "canonical_family"
].apply(
    normalize_family
)

approved["canonical_id"] = pd.to_numeric(
    approved["canonical_id"],
    errors="coerce"
)


# ============================================================
# REMOVE INVALID APPROVED MAPPINGS
# ============================================================

approved = approved[
    (approved["description_clean"] != "")
    &
    (approved["vch_type_clean"] != "")
    &
    (approved["canonical_id"].notna())
    &
    (approved["canonical_family"] != "")
].copy()


# ============================================================
# CHECK CONFLICTING APPROVED MAPPINGS
# ============================================================

mapping_conflicts = (
    approved
    .groupby(
        [
            "description_clean",
            "vch_type_clean",
        ]
    )["canonical_id"]
    .nunique()
)

conflicts = mapping_conflicts[
    mapping_conflicts > 1
]


if len(conflicts) > 0:

    raise ValueError(
        "Conflicting approved mappings detected: "
        f"{len(conflicts)}"
    )


# ============================================================
# REMOVE DUPLICATE APPROVED MAPPING KEYS
# ============================================================

approved = approved.drop_duplicates(
    subset=[
        "description_clean",
        "vch_type_clean",
    ]
).copy()


print(
    f"Approved mapping rules loaded: "
    f"{len(approved):,}"
)


# ============================================================
# APPLY APPROVED DESCRIPTION MAPPINGS
# ============================================================

approved_lookup = approved.set_index(
    [
        "description_clean",
        "vch_type_clean",
    ]
)[
    [
        "canonical_id",
        "canonical_family",
    ]
].to_dict(
    "index"
)


approved_count = 0


for idx, row in df.iterrows():

    key = (
        row["description_clean"],
        row["vch_type_clean"],
    )

    if key in approved_lookup:

        result = approved_lookup[key]

        df.at[
            idx,
            "canonical_id"
        ] = result["canonical_id"]

        df.at[
            idx,
            "canonical_family"
        ] = result["canonical_family"]

        df.at[
            idx,
            "canonical_mapping_status"
        ] = "MAPPED"

        df.at[
            idx,
            "mapping_method"
        ] = "APPROVED_MAPPING"

        approved_count += 1


print(
    f"Approved mappings applied: "
    f"{approved_count:,}"
)


# ============================================================
# APPLY DETERMINISTIC DESCRIPTION RULES
# ============================================================

rule_count = 0


for idx, row in df.iterrows():

    # Never overwrite an existing mapping.

    if (
        df.at[
            idx,
            "canonical_family"
        ]
        != "UNMAPPED"
    ):
        continue

    family = apply_deterministic_rule(
        row["description_clean"]
    )

    if family is not None:

        canonical_id = family_to_canonical_id(
            family
        )

        df.at[
            idx,
            "canonical_id"
        ] = canonical_id

        df.at[
            idx,
            "canonical_family"
        ] = family

        df.at[
            idx,
            "canonical_mapping_status"
        ] = "MAPPED"

        df.at[
            idx,
            "mapping_method"
        ] = "RULE"

        rule_count += 1


print(
    f"Deterministic rules mapped: "
    f"{rule_count:,}"
)


# ============================================================
# LOAD STRONG CATEGORY_ID MAPPINGS
# ============================================================

print()
print("=" * 90)
print("LOADING STRONG CATEGORY_ID MAPPINGS")
print("=" * 90)


category_analysis = pd.read_csv(
    CATEGORY_MAPPING_PATH
)


required_category_columns = [
    "category_id_clean",
    "dominant_canonical_family",
    "dominant_count",
    "mapped_total",
    "dominant_percentage",
]


missing_category_columns = [
    col
    for col in required_category_columns
    if col not in category_analysis.columns
]


if missing_category_columns:

    raise ValueError(
        "Missing columns in category ID analysis: "
        f"{missing_category_columns}"
    )


category_analysis[
    "category_id_clean"
] = category_analysis[
    "category_id_clean"
].apply(
    normalize_category_id
)


category_analysis[
    "dominant_canonical_family"
] = category_analysis[
    "dominant_canonical_family"
].apply(
    normalize_family
)


# ============================================================
# SELECT ONLY HIGH-CONFIDENCE CATEGORY_ID MAPPINGS
# ============================================================

strong_category_mappings = category_analysis[
    (
        category_analysis[
            "dominant_percentage"
        ]
        >= CATEGORY_ID_MIN_PERCENTAGE
    )
    &
    (
        category_analysis[
            "mapped_total"
        ]
        >= CATEGORY_ID_MIN_MAPPED_COUNT
    )
].copy()


# ============================================================
# KEEP ONLY CANONICAL TAXONOMY FAMILIES
# ============================================================

strong_category_mappings = (
    strong_category_mappings[
        strong_category_mappings[
            "dominant_canonical_family"
        ].isin(
            CANONICAL_FAMILIES.values()
        )
    ]
)


# ============================================================
# BUILD CATEGORY_ID LOOKUP
# ============================================================

category_lookup = dict(
    zip(
        strong_category_mappings[
            "category_id_clean"
        ],
        strong_category_mappings[
            "dominant_canonical_family"
        ],
    )
)


print(
    f"Strong category_id mappings loaded: "
    f"{len(category_lookup):,}"
)

print(
    f"Criteria: >= "
    f"{CATEGORY_ID_MIN_PERCENTAGE:.0f}% dominance "
    f"and >= "
    f"{CATEGORY_ID_MIN_MAPPED_COUNT} mapped examples"
)


# ============================================================
# SHOW STRONG CATEGORY_ID MAPPINGS
# ============================================================

if len(strong_category_mappings) > 0:

    print()
    print("Strong category_id rules:")

    display_columns = [
        "category_id_clean",
        "dominant_canonical_family",
        "dominant_count",
        "mapped_total",
        "dominant_percentage",
    ]

    print(
        strong_category_mappings[
            display_columns
        ]
        .sort_values(
            by="category_id_clean"
        )
        .to_string(
            index=False
        )
    )


# ============================================================
# APPLY STRONG CATEGORY_ID MAPPINGS
# ============================================================

category_id_count = 0


for idx, row in df.iterrows():

    # Never overwrite an existing mapping.

    if (
        df.at[
            idx,
            "canonical_family"
        ]
        != "UNMAPPED"
    ):
        continue

    category_id = row[
        "category_id_clean"
    ]

    if category_id in category_lookup:

        family = category_lookup[
            category_id
        ]

        canonical_id = family_to_canonical_id(
            family
        )

        df.at[
            idx,
            "canonical_id"
        ] = canonical_id

        df.at[
            idx,
            "canonical_family"
        ] = family

        df.at[
            idx,
            "canonical_mapping_status"
        ] = "MAPPED"

        df.at[
            idx,
            "mapping_method"
        ] = "CATEGORY_ID"

        category_id_count += 1


print(
    f"Strong category_id mappings applied: "
    f"{category_id_count:,}"
)


# ============================================================
# LOAD ML MODEL
# ============================================================

print()
print("=" * 90)
print("ML CLASSIFICATION")
print("=" * 90)


unresolved_mask = (
    df["canonical_family"]
    == "UNMAPPED"
)


ml_candidates = df.loc[
    unresolved_mask
].copy()


# Only non-empty descriptions.

ml_candidates = ml_candidates[
    ml_candidates[
        "description_clean"
    ] != ""
].copy()


print(
    f"ML candidates: "
    f"{len(ml_candidates):,}"
)


ml_accepted = 0


if len(ml_candidates) > 0:

    # ========================================================
    # TRY LOADING PREVIOUSLY TRAINED MODEL
    # ========================================================

    model = None


    if MODEL_PATH.exists():

        try:

            model = joblib.load(
                MODEL_PATH
            )

            print(
                f"Loaded ML model: "
                f"{MODEL_PATH}"
            )

        except Exception as e:

            print(
                "Could not load saved model."
            )

            print(e)


    # ========================================================
    # TRAIN MODEL IF SAVED MODEL IS UNAVAILABLE
    # ========================================================

    if model is None:

        training_path = (
            ROOT
            / "outputs"
            / "ml_training_dataset.csv"
        )


        if not training_path.exists():

            raise FileNotFoundError(
                "ML model and training dataset "
                "are both unavailable."
            )


        training_df = pd.read_csv(
            training_path
        )


        training_df[
            "canonical_family"
        ] = training_df[
            "canonical_family"
        ].apply(
            normalize_family
        )


        training_df[
            "vch_desc"
        ] = training_df[
            "vch_desc"
        ].apply(
            normalize_description
        )


        training_df = training_df[
            (
                training_df[
                    "vch_desc"
                ] != ""
            )
            &
            (
                training_df[
                    "canonical_family"
                ].isin(
                    CANONICAL_FAMILIES.values()
                )
            )
        ].copy()


        # ====================================================
        # TF-IDF
        # ====================================================

        vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.98,
            sublinear_tf=True,
        )


        X_train = vectorizer.fit_transform(
            training_df[
                "vch_desc"
            ]
        )


        # ====================================================
        # LOGISTIC REGRESSION
        # ====================================================

        classifier = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        )


        classifier.fit(
            X_train,
            training_df[
                "canonical_family"
            ]
        )


        model = {
            "vectorizer": vectorizer,
            "classifier": classifier,
        }


    # ========================================================
    # SUPPORT DICTIONARY AND SKLEARN PIPELINE MODELS
    # ========================================================

    if isinstance(
        model,
        dict
    ):

        vectorizer = model[
            "vectorizer"
        ]

        classifier = model[
            "classifier"
        ]


        X_ml = vectorizer.transform(
            ml_candidates[
                "description_clean"
            ]
        )


        probabilities = (
            classifier.predict_proba(
                X_ml
            )
        )


        classes = classifier.classes_


    else:

        probabilities = (
            model.predict_proba(
                ml_candidates[
                    "description_clean"
                ]
            )
        )


        classes = model.classes_


    # ========================================================
    # TOP PREDICTION
    # ========================================================

    top_indices = np.argmax(
        probabilities,
        axis=1
    )


    top_confidences = probabilities[
        np.arange(
            len(probabilities)
        ),
        top_indices
    ]


    sorted_probabilities = np.sort(
        probabilities,
        axis=1
    )


    if probabilities.shape[1] >= 2:

        second_confidences = (
            sorted_probabilities[
                :,
                -2
            ]
        )

    else:

        second_confidences = np.zeros(
            len(probabilities)
        )


    # ========================================================
    # CONFIDENCE MARGIN
    # ========================================================

    margins = (
        top_confidences
        - second_confidences
    )


    # ========================================================
    # PREDICTED FAMILIES
    # ========================================================

    predicted_families = [
        classes[i]
        for i in top_indices
    ]


    ml_candidates[
        "predicted_family"
    ] = predicted_families


    ml_candidates[
        "ml_confidence"
    ] = top_confidences


    ml_candidates[
        "ml_margin"
    ] = margins


    # ========================================================
    # APPLY ML THRESHOLDS
    # ========================================================

    accepted_mask = (
        (
            ml_candidates[
                "ml_confidence"
            ]
            >= ML_CONFIDENCE_THRESHOLD
        )
        &
        (
            ml_candidates[
                "ml_margin"
            ]
            >= ML_MARGIN_THRESHOLD
        )
    )


    accepted = ml_candidates[
        accepted_mask
    ].copy()


    # ========================================================
    # APPLY ACCEPTED ML PREDICTIONS
    # ========================================================

    for idx, row in accepted.iterrows():

        family = row[
            "predicted_family"
        ]


        canonical_id = family_to_canonical_id(
            family
        )


        df.at[
            idx,
            "canonical_id"
        ] = canonical_id


        df.at[
            idx,
            "canonical_family"
        ] = family


        df.at[
            idx,
            "canonical_mapping_status"
        ] = "MAPPED"


        df.at[
            idx,
            "mapping_method"
        ] = "ML"


        df.at[
            idx,
            "ml_confidence"
        ] = row[
            "ml_confidence"
        ]


        df.at[
            idx,
            "ml_margin"
        ] = row[
            "ml_margin"
        ]


        ml_accepted += 1


print(
    f"ML accepted: "
    f"{ml_accepted:,}"
)


print(
    f"ML rejected: "
    f"{len(ml_candidates) - ml_accepted:,}"
)


# ============================================================
# FORCE CANONICAL FAMILY FROM CANONICAL ID
# ============================================================

for canonical_id, family in CANONICAL_FAMILIES.items():

    mask = (
        pd.to_numeric(
            df[
                "canonical_id"
            ],
            errors="coerce"
        )
        == canonical_id
    )


    df.loc[
        mask,
        "canonical_family"
    ] = family


# ============================================================
# FINAL CLEANUP
# ============================================================

mapped_mask = (
    df[
        "canonical_family"
    ]
    != "UNMAPPED"
)


df.loc[
    mapped_mask,
    "canonical_mapping_status"
] = "MAPPED"


# ============================================================
# VALIDATION
# ============================================================

invalid_families = sorted(
    set(
        df[
            "canonical_family"
        ].dropna()
    )
    -
    set(
        CANONICAL_FAMILIES.values()
    )
    -
    {
        "UNMAPPED"
    }
)


if invalid_families:

    raise ValueError(
        "Invalid canonical families found: "
        f"{invalid_families}"
    )


# ============================================================
# SUMMARY
# ============================================================

automatically_mapped = int(
    (
        df[
            "canonical_family"
        ]
        != "UNMAPPED"
    ).sum()
)


unmapped = int(
    (
        df[
            "canonical_family"
        ]
        == "UNMAPPED"
    ).sum()
)


coverage = (
    automatically_mapped
    / len(df)
    * 100
)


print()
print("=" * 90)
print("FINAL MAPPING SUMMARY")
print("=" * 90)


print(
    f"Automatically mapped: "
    f"{automatically_mapped:,}"
)


print(
    f"Unmapped: "
    f"{unmapped:,}"
)


print(
    f"Coverage: "
    f"{coverage:.2f}%"
)


# ============================================================
# METHOD BREAKDOWN
# ============================================================

print()
print("=" * 90)
print("MAPPING METHOD BREAKDOWN")
print("=" * 90)


method_counts = (
    df[
        "mapping_method"
    ]
    .value_counts()
)


print(
    method_counts.to_string()
)


# ============================================================
# CATEGORY BREAKDOWN
# ============================================================

print()
print("=" * 90)
print("CANONICAL CATEGORY BREAKDOWN")
print("=" * 90)


category_counts = (
    df[
        "canonical_family"
    ]
    .value_counts()
)


print(
    category_counts.to_string()
)


# ============================================================
# CATEGORY_ID MAPPING SUMMARY
# ============================================================

print()
print("=" * 90)
print("CATEGORY_ID MAPPING SUMMARY")
print("=" * 90)


print(
    f"Strong category_id rules available: "
    f"{len(category_lookup):,}"
)


print(
    f"Transactions mapped using category_id: "
    f"{category_id_count:,}"
)


# ============================================================
# ACCEPTED ML CONFIDENCE SUMMARY
# ============================================================

accepted_ml_rows = df[
    df[
        "mapping_method"
    ]
    == "ML"
].copy()


if len(accepted_ml_rows) > 0:

    print()
    print("=" * 90)
    print("ACCEPTED ML CONFIDENCE SUMMARY")
    print("=" * 90)


    print(
        f"Accepted ML transactions: "
        f"{len(accepted_ml_rows):,}"
    )


    print(
        f"Average confidence: "
        f"{accepted_ml_rows['ml_confidence'].mean():.4f}"
    )


    print(
        f"Minimum confidence: "
        f"{accepted_ml_rows['ml_confidence'].min():.4f}"
    )


    print(
        f"Maximum confidence: "
        f"{accepted_ml_rows['ml_confidence'].max():.4f}"
    )


    print(
        f"Average margin: "
        f"{accepted_ml_rows['ml_margin'].mean():.4f}"
    )


# ============================================================
# REAL-WORLD UNRESOLVED ML DIAGNOSTICS
# ============================================================

if len(ml_candidates) > 0:

    print()
    print("=" * 90)
    print(
        "REAL-WORLD UNRESOLVED ML "
        "CONFIDENCE DISTRIBUTION"
    )
    print("=" * 90)


    confidence = ml_candidates[
        "ml_confidence"
    ]


    margin = ml_candidates[
        "ml_margin"
    ]


    print(
        f"Total unresolved transactions "
        f"evaluated by ML: "
        f"{len(ml_candidates):,}"
    )


    print(
        f"Min confidence: "
        f"{confidence.min():.4f}"
    )


    print(
        f"25th percentile: "
        f"{confidence.quantile(0.25):.4f}"
    )


    print(
        f"Median confidence: "
        f"{confidence.median():.4f}"
    )


    print(
        f"75th percentile: "
        f"{confidence.quantile(0.75):.4f}"
    )


    print(
        f"Average confidence: "
        f"{confidence.mean():.4f}"
    )


    print(
        f"Max confidence: "
        f"{confidence.max():.4f}"
    )


    print(
        f"Min margin: "
        f"{margin.min():.4f}"
    )


    print(
        f"Median margin: "
        f"{margin.median():.4f}"
    )


    print(
        f"Average margin: "
        f"{margin.mean():.4f}"
    )


    # ========================================================
    # CONFIDENCE THRESHOLD DISTRIBUTION
    # ========================================================

    for threshold in [
        0.50,
        0.60,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.95,
    ]:

        count = int(
            (
                confidence
                >= threshold
            ).sum()
        )


        percentage = (
            count
            / len(ml_candidates)
            * 100
        )


        print(
            f">= {threshold:.2f}: "
            f"{count:,} "
            f"({percentage:.2f}%)"
        )


# ============================================================
# SAVE OUTPUT
# ============================================================

df.to_csv(
    OUTPUT_PATH,
    index=False
)


print(
    OUTPUT_PATH
)
