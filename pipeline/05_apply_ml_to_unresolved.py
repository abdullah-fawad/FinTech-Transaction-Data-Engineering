import os
from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "hk_transactions_with_refined_rules.csv"
)

MODEL_FILE = (
    BASE_DIR
    / "outputs"
    / "ml_model"
    / "transaction_category_classifier.joblib"
)

OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
    / "ml_predictions"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "unresolved_ml_predictions.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "unresolved_ml_predictions_summary.csv"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 90)
print("APPLYING ML MODEL TO UNRESOLVED TRANSACTIONS")
print("=" * 90)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD TRANSACTIONS
# ============================================================

print()
print("Loading transactions:")
print(INPUT_FILE)

df = pd.read_csv(INPUT_FILE)

print(f"Transactions loaded: {len(df):,}")


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "vch_desc",
    "vch_type",
    "category_id",
    "refined_rule_method"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# CLEAN TRANSACTION DESCRIPTION
# ============================================================

df["ml_description"] = (
    df["vch_desc"]
    .fillna("")
    .astype(str)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)


# ============================================================
# IDENTIFY TRANSACTIONS ALREADY HANDLED
# ============================================================

print()
print("=" * 90)
print("IDENTIFYING UNRESOLVED TRANSACTIONS")
print("=" * 90)


trusted_shopping_methods = [
    "REFINED_SHOPPING_EXPENSE",
    "REFINED_SHOPPING_TRANSFER",
    "REFINED_SHOPPING_INCOME"
]


already_classified = (
    df["refined_rule_method"]
    .fillna("")
    .astype(str)
    .isin(trusted_shopping_methods)
)


# ============================================================
# CREATE ML CANDIDATE POOL
# ============================================================

candidate_mask = (
    ~already_classified
    & df["ml_description"].ne("")
)

candidates = df.loc[candidate_mask].copy()


print()
print(
    f"Already handled by trusted Shopping rules: "
    f"{already_classified.sum():,}"
)

print(
    f"Unresolved ML candidates: "
    f"{len(candidates):,}"
)


# ============================================================
# SAFETY CHECK
# ============================================================

if len(candidates) == 0:
    raise ValueError(
        "No unresolved ML candidates were found."
    )


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

print()
print("=" * 90)
print("LOADING TRAINED ML MODEL")
print("=" * 90)

print()
print("Model:")
print(MODEL_FILE)

if not MODEL_FILE.exists():
    raise FileNotFoundError(
        f"Model file not found:\n{MODEL_FILE}"
    )

model = joblib.load(MODEL_FILE)

print("Model loaded successfully.")


# ============================================================
# MODEL INFORMATION
# ============================================================

print()
print("=" * 90)
print("MODEL INFORMATION")
print("=" * 90)

try:
    classifier = model.named_steps["classifier"]

    print()
    print("Model classes:")

    for class_name in classifier.classes_:
        print(f"  - {class_name}")

    print()
    print(f"Number of classes: {len(classifier.classes_)}")

except Exception:
    print(
        "Model loaded, but model class information "
        "could not be displayed."
    )


# ============================================================
# RUN ML PREDICTIONS
# ============================================================

print()
print("=" * 90)
print("RUNNING ML PREDICTIONS")
print("=" * 90)

texts = candidates["ml_description"].tolist()

predictions = model.predict(texts)

probabilities = model.predict_proba(texts)


# ============================================================
# CALCULATE TOP PREDICTION CONFIDENCE
# ============================================================

confidence = probabilities.max(axis=1)

candidates["ml_prediction"] = predictions

candidates["ml_confidence"] = confidence


# ============================================================
# CALCULATE SECOND-BEST PREDICTION
# ============================================================

classes = model.classes_

second_prediction = []
second_confidence = []

for row in probabilities:

    sorted_indices = row.argsort()[::-1]

    # Best prediction is index 0.
    # We take index 1 as the second-best prediction.

    if len(sorted_indices) > 1:

        second_index = sorted_indices[1]

        second_prediction.append(
            classes[second_index]
        )

        second_confidence.append(
            row[second_index]
        )

    else:

        second_prediction.append(
            classes[sorted_indices[0]]
        )

        second_confidence.append(
            row[sorted_indices[0]]
        )


candidates["ml_second_prediction"] = second_prediction

candidates["ml_second_confidence"] = second_confidence


# ============================================================
# CONFIDENCE GAP
# ============================================================

# Example:
#
# Shopping = 0.90
# Dining   = 0.07
#
# Confidence gap = 0.83
#
# A larger gap generally means the model is more decisive.

candidates["ml_confidence_gap"] = (
    candidates["ml_confidence"]
    - candidates["ml_second_confidence"]
)


# ============================================================
# CONFIDENCE BUCKET
# ============================================================

def confidence_bucket(confidence):

    if confidence >= 0.90:
        return "HIGH_CONFIDENCE"

    elif confidence >= 0.70:
        return "MEDIUM_CONFIDENCE"

    elif confidence >= 0.50:
        return "LOW_CONFIDENCE"

    else:
        return "VERY_LOW_CONFIDENCE"


candidates["ml_confidence_bucket"] = (
    candidates["ml_confidence"]
    .apply(confidence_bucket)
)


# ============================================================
# ORGANIZE OUTPUT COLUMNS
# ============================================================

preferred_columns = [
    "vch_desc",
    "ml_description",
    "vch_type",
    "category_id",

    "refined_rule_method",
    "rule_matched_term",

    "ml_prediction",
    "ml_confidence",

    "ml_second_prediction",
    "ml_second_confidence",

    "ml_confidence_gap",
    "ml_confidence_bucket"
]


available_columns = [
    column
    for column in preferred_columns
    if column in candidates.columns
]


remaining_columns = [
    column
    for column in candidates.columns
    if column not in available_columns
]


candidates = candidates[
    available_columns + remaining_columns
]


# ============================================================
# SAVE PREDICTIONS
# ============================================================

candidates.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# CREATE SUMMARY
# ============================================================

summary = (
    candidates["ml_prediction"]
    .value_counts()
    .rename_axis("ml_prediction")
    .reset_index(name="count")
)

summary["percentage"] = (
    summary["count"]
    / len(candidates)
    * 100
)

summary.to_csv(
    SUMMARY_FILE,
    index=False
)


# ============================================================
# DISPLAY PREDICTED FAMILY DISTRIBUTION
# ============================================================

print()
print("=" * 90)
print("PREDICTED FAMILY DISTRIBUTION")
print("=" * 90)

print(
    candidates["ml_prediction"]
    .value_counts()
)


# ============================================================
# DISPLAY CONFIDENCE DISTRIBUTION
# ============================================================

print()
print("=" * 90)
print("CONFIDENCE DISTRIBUTION")
print("=" * 90)

print(
    candidates["ml_confidence"]
    .describe()
)


# ============================================================
# CONFIDENCE BUCKETS
# ============================================================

print()
print("=" * 90)
print("CONFIDENCE BUCKETS")
print("=" * 90)

bucket_counts = (
    candidates["ml_confidence_bucket"]
    .value_counts()
)

print(bucket_counts)


# ============================================================
# CONFIDENCE PERCENTAGES
# ============================================================

print()
print("=" * 90)
print("CONFIDENCE PERCENTAGES")
print("=" * 90)

bucket_percentages = (
    candidates["ml_confidence_bucket"]
    .value_counts(normalize=True)
    * 100
)

print(
    bucket_percentages.round(2)
)


# ============================================================
# HIGH-CONFIDENCE SAMPLE
# ============================================================

print()
print("=" * 90)
print("SAMPLE HIGH-CONFIDENCE PREDICTIONS")
print("=" * 90)

high_confidence = (
    candidates[
        candidates["ml_confidence"] >= 0.90
    ]
    [
        [
            "vch_desc",
            "vch_type",
            "ml_prediction",
            "ml_confidence"
        ]
    ]
    .head(30)
)

if len(high_confidence) > 0:

    print(
        high_confidence.to_string(index=False)
    )

else:

    print("No high-confidence predictions found.")


# ============================================================
# LOW-CONFIDENCE SAMPLE
# ============================================================

print()
print("=" * 90)
print("SAMPLE LOW-CONFIDENCE PREDICTIONS")
print("=" * 90)

low_confidence = (
    candidates
    .sort_values("ml_confidence")
    [
        [
            "vch_desc",
            "vch_type",
            "ml_prediction",
            "ml_confidence",
            "ml_second_prediction",
            "ml_second_confidence",
            "ml_confidence_gap"
        ]
    ]
    .head(30)
)

if len(low_confidence) > 0:

    print(
        low_confidence.to_string(index=False)
    )

else:

    print("No predictions found.")


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 90)
print("ML APPLICATION COMPLETE")
print("=" * 90)

total_candidates = len(candidates)

high_count = (
    candidates["ml_confidence"] >= 0.90
).sum()

medium_count = (
    (candidates["ml_confidence"] >= 0.70)
    &
    (candidates["ml_confidence"] < 0.90)
).sum()

low_count = (
    (candidates["ml_confidence"] >= 0.50)
    &
    (candidates["ml_confidence"] < 0.70)
).sum()

very_low_count = (
    candidates["ml_confidence"] < 0.50
).sum()


print()
print(
    f"Total transactions:              {len(df):,}"
)

print(
    f"Already handled:                  "
    f"{already_classified.sum():,}"
)

print(
    f"ML candidates:                    "
    f"{total_candidates:,}"
)

print(
    f"High confidence (>= 0.90):       "
    f"{high_count:,}"
)

print(
    f"Medium confidence (0.70-0.89):   "
    f"{medium_count:,}"
)

print(
    f"Low confidence (0.50-0.69):      "
    f"{low_count:,}"
)

print(
    f"Very low confidence (< 0.50):    "
    f"{very_low_count:,}"
)


# ============================================================
# FILE LOCATIONS
# ============================================================

print()
print("=" * 90)
print("FILES CREATED")
print("=" * 90)

print()
print("Prediction file:")
print(OUTPUT_FILE)

print()
print("Summary file:")
print(SUMMARY_FILE)
