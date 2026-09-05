import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import joblib


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

TRAINING_FILE = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "ml_training_dataset.csv"
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models",
    "ml_model"
)

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "transaction_category_classifier.joblib"
)

REPORT_FILE = os.path.join(
    MODEL_DIR,
    "classification_report.txt"
)

CONFUSION_FILE = os.path.join(
    MODEL_DIR,
    "confusion_matrix.csv"
)

THRESHOLD_FILE = os.path.join(
    MODEL_DIR,
    "threshold_evaluation.csv"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# THRESHOLD SETTINGS TO TEST
# ============================================================

CONFIDENCE_THRESHOLDS = [
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95
]

MARGIN_THRESHOLDS = [
    0.00,
    0.05,
    0.10,
    0.15,
    0.20,
    0.25
]


# ============================================================
# HEADER
# ============================================================

print("=" * 90)
print("TRAINING ML TRANSACTION CLASSIFIER")
print("=" * 90)


# ============================================================
# LOAD TRAINING DATA
# ============================================================

print()
print("Loading training dataset:")
print(TRAINING_FILE)

if not os.path.exists(TRAINING_FILE):

    raise FileNotFoundError(
        f"Training dataset not found:\n{TRAINING_FILE}"
    )

df = pd.read_csv(
    TRAINING_FILE
)

print(
    f"Training examples loaded: "
    f"{len(df):,}"
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "vch_desc",
    "canonical_family",
    "canonical_id"
]

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:

    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(
            f"- {col}"
            for col in missing_columns
        )
    )


# ============================================================
# PREPARE TRAINING DATA
# ============================================================

print()
print("=" * 90)
print("PREPARING TRAINING DATA")
print("=" * 90)


df["ml_description"] = (
    df["vch_desc"]
    .fillna("")
    .astype(str)
    .str.strip()
)


df["training_family"] = (
    df["canonical_family"]
    .fillna("")
    .astype(str)
    .str.strip()
)


# ------------------------------------------------------------
# Normalize older category-family names
# into the current canonical taxonomy.
# ------------------------------------------------------------

FAMILY_NORMALIZATION = {

    "dining": "Food & Groceries",

    "groceries": "Food & Groceries",

    "food": "Food & Groceries",

    "food & groceries": "Food & Groceries",

    "fuel": "Fuel",

    "transport": "Transport",

    "utilities": "Utilities & Bills",

    "utilities & bills": "Utilities & Bills",

    "healthcare": "Healthcare",

    "shopping": "Shopping & Retail",

    "shopping & retail": "Shopping & Retail",

    "transfers": "Transfers & Withdrawals",

    "transfers & withdrawals":
        "Transfers & Withdrawals",

    "income": "Income",

    "other": "Other"
}


def normalize_family(value):

    value = str(value).strip()

    key = value.lower()

    return FAMILY_NORMALIZATION.get(
        key,
        value
    )


df["training_family"] = (
    df["training_family"]
    .apply(normalize_family)
)


# ------------------------------------------------------------
# Remove empty records
# ------------------------------------------------------------

before = len(df)

df = df[
    (df["ml_description"] != "")
    &
    (df["training_family"] != "")
].copy()

removed = before - len(df)

print(
    f"Removed empty training records: "
    f"{removed:,}"
)

print(
    f"Final usable training examples: "
    f"{len(df):,}"
)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

class_counts = (
    df["training_family"]
    .value_counts()
    .sort_values(ascending=False)
)

print()
print("=" * 90)
print("TARGET CLASS DISTRIBUTION")
print("=" * 90)

print(class_counts)

print()
print(
    f"Number of classes: "
    f"{df['training_family'].nunique()}"
)


classes_with_less_than_two = (
    class_counts[
        class_counts < 2
    ]
)

if len(classes_with_less_than_two) > 0:

    raise ValueError(
        "Some classes contain fewer than 2 records:\n"
        + "\n".join(
            f"- {idx}: {count}"
            for idx, count
            in classes_with_less_than_two.items()
        )
    )


# ============================================================
# FEATURES AND TARGET
# ============================================================

X = df["ml_description"]

y = df["training_family"]


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

print()
print("=" * 90)
print("CREATING TRAIN / TEST SPLIT")
print("=" * 90)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(
    f"Training records: "
    f"{len(X_train):,}"
)

print(
    f"Testing records: "
    f"{len(X_test):,}"
)


# ============================================================
# BUILD MODEL
# ============================================================

print()
print("=" * 90)
print("BUILDING ML PIPELINE")
print("=" * 90)

print("Vectorizer: TF-IDF")
print("Classifier: Logistic Regression")


model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.98,
            sublinear_tf=True
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        )
    )
])


# ============================================================
# TRAIN MODEL
# ============================================================

print()
print("=" * 90)
print("TRAINING MODEL")
print("=" * 90)

model.fit(
    X_train,
    y_train
)

print(
    "Model training completed."
)


# ============================================================
# PREDICTIONS
# ============================================================

print()
print("=" * 90)
print("GENERATING TEST PREDICTIONS")
print("=" * 90)

y_pred = model.predict(
    X_test
)

y_proba = model.predict_proba(
    X_test
)


# ============================================================
# PROBABILITY INFORMATION
# ============================================================

top1_probability = np.max(
    y_proba,
    axis=1
)

sorted_probabilities = np.sort(
    y_proba,
    axis=1
)[:, ::-1]


if y_proba.shape[1] >= 2:

    top2_probability = (
        sorted_probabilities[:, 1]
    )

else:

    top2_probability = np.zeros(
        len(y_proba)
    )


margin = (
    top1_probability
    - top2_probability
)


# ============================================================
# BASIC ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print()
print(
    f"Overall test accuracy: "
    f"{accuracy * 100:.2f}%"
)


# ============================================================
# THRESHOLD EVALUATION
# ============================================================

print()
print("=" * 90)
print("THRESHOLD EVALUATION")
print("=" * 90)

threshold_results = []

for confidence_threshold in (
    CONFIDENCE_THRESHOLDS
):

    for margin_threshold in (
        MARGIN_THRESHOLDS
    ):

        accepted = (
            (top1_probability >= confidence_threshold)
            &
            (margin >= margin_threshold)
        )

        accepted_count = int(
            accepted.sum()
        )

        coverage = (
            accepted_count
            / len(y_test)
        )

        if accepted_count > 0:

            accepted_accuracy = (
                y_pred[accepted]
                == y_test.values[accepted]
            ).mean()

        else:

            accepted_accuracy = np.nan


        threshold_results.append({

            "confidence_threshold":
                confidence_threshold,

            "margin_threshold":
                margin_threshold,

            "accepted_predictions":
                accepted_count,

            "coverage_pct":
                round(
                    coverage * 100,
                    2
                ),

            "accepted_accuracy_pct":
                round(
                    accepted_accuracy * 100,
                    2
                )
                if not np.isnan(
                    accepted_accuracy
                )
                else None
        })


threshold_df = pd.DataFrame(
    threshold_results
)


# ============================================================
# PRINT THRESHOLD RESULTS
# ============================================================

print()

print(
    f"{'Confidence':<15}"
    f"{'Margin':<12}"
    f"{'Accepted':<12}"
    f"{'Coverage %':<14}"
    f"{'Accuracy %'}"
)

print("-" * 70)

for _, row in threshold_df.iterrows():

    accuracy_text = (
        f"{row['accepted_accuracy_pct']:.2f}"
        if pd.notna(
            row["accepted_accuracy_pct"]
        )
        else "N/A"
    )

    print(
        f"{row['confidence_threshold']:<15.2f}"
        f"{row['margin_threshold']:<12.2f}"
        f"{row['accepted_predictions']:<12}"
        f"{row['coverage_pct']:<14.2f}"
        f"{accuracy_text}"
    )


# ============================================================
# RECOMMENDED THRESHOLDS
# ============================================================

print()
print("=" * 90)
print("THRESHOLD RECOMMENDATIONS")
print("=" * 90)


# We prefer high precision first.
#
# The target is at least 95% accuracy among
# automatically accepted predictions.
#
# Within that constraint, choose the combination
# with the highest coverage.

VALID_TARGET = 95.0

valid_thresholds = threshold_df[
    (
        threshold_df[
            "accepted_accuracy_pct"
        ].notna()
    )
    &
    (
        threshold_df[
            "accepted_accuracy_pct"
        ]
        >= VALID_TARGET
    )
].copy()


if len(valid_thresholds) > 0:

    recommended = (
        valid_thresholds
        .sort_values(
            [
                "coverage_pct",
                "accepted_accuracy_pct"
            ],
            ascending=[
                False,
                False
            ]
        )
        .iloc[0]
    )

    print()
    print(
        "Recommended combination "
        "(highest coverage with >= 95% test accuracy):"
    )

    print(
        f"Confidence threshold: "
        f"{recommended['confidence_threshold']:.2f}"
    )

    print(
        f"Margin threshold: "
        f"{recommended['margin_threshold']:.2f}"
    )

    print(
        f"Accepted predictions: "
        f"{int(recommended['accepted_predictions']):,}"
    )

    print(
        f"Coverage: "
        f"{recommended['coverage_pct']:.2f}%"
    )

    print(
        f"Accepted prediction accuracy: "
        f"{recommended['accepted_accuracy_pct']:.2f}%"
    )

else:

    print()
    print(
        "No tested threshold combination "
        "reached 95% accepted accuracy."
    )

    # If 95% cannot be achieved, show the
    # highest-accuracy combination instead.

    fallback = (
        threshold_df
        .dropna(
            subset=[
                "accepted_accuracy_pct"
            ]
        )
        .sort_values(
            [
                "accepted_accuracy_pct",
                "coverage_pct"
            ],
            ascending=[
                False,
                False
            ]
        )
        .iloc[0]
    )

    print()
    print(
        "Best available combination:"
    )

    print(
        f"Confidence threshold: "
        f"{fallback['confidence_threshold']:.2f}"
    )

    print(
        f"Margin threshold: "
        f"{fallback['margin_threshold']:.2f}"
    )

    print(
        f"Accepted predictions: "
        f"{int(fallback['accepted_predictions']):,}"
    )

    print(
        f"Coverage: "
        f"{fallback['coverage_pct']:.2f}%"
    )

    print(
        f"Accepted prediction accuracy: "
        f"{fallback['accepted_accuracy_pct']:.2f}%"
    )


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_test,
    y_pred,
    zero_division=0
)

print()
print("=" * 90)
print("CLASSIFICATION REPORT")
print("=" * 90)

print(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

labels = sorted(
    y.unique()
)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

confusion_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels
)

confusion_df.index.name = "actual"

confusion_df.to_csv(
    CONFUSION_FILE
)


# ============================================================
# SAVE CLASSIFICATION REPORT
# ============================================================

with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "TRANSACTION CATEGORY CLASSIFIER\n"
    )

    file.write(
        "=" * 70
        + "\n\n"
    )

    file.write(
        f"Training examples: "
        f"{len(X_train):,}\n"
    )

    file.write(
        f"Testing examples: "
        f"{len(X_test):,}\n"
    )

    file.write(
        f"Number of classes: "
        f"{len(labels)}\n"
    )

    file.write(
        f"Overall accuracy: "
        f"{accuracy:.4f}\n"
    )

    file.write(
        f"Overall accuracy percentage: "
        f"{accuracy * 100:.2f}%\n\n"
    )

    file.write(
        "CLASSIFICATION REPORT\n"
    )

    file.write(
        "=" * 70
        + "\n"
    )

    file.write(report)


# ============================================================
# SAVE THRESHOLD RESULTS
# ============================================================

threshold_df.to_csv(
    THRESHOLD_FILE,
    index=False
)


# ============================================================
# CONFIDENCE SUMMARY
# ============================================================

print()
print("=" * 90)
print("PREDICTION CONFIDENCE")
print("=" * 90)

print(
    f"Average confidence: "
    f"{top1_probability.mean():.4f}"
)

print(
    f"Minimum confidence: "
    f"{top1_probability.min():.4f}"
)

print(
    f"Maximum confidence: "
    f"{top1_probability.max():.4f}"
)

print(
    f"Average top-1/top-2 margin: "
    f"{margin.mean():.4f}"
)


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    model,
    MODEL_FILE
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 90)
print("MODEL TRAINING COMPLETE")
print("=" * 90)

print()
print("Model:")
print(MODEL_FILE)

print()
print("Classification report:")
print(REPORT_FILE)

print()
print("Confusion matrix:")
print(CONFUSION_FILE)

print()
print("Threshold evaluation:")
print(THRESHOLD_FILE)

