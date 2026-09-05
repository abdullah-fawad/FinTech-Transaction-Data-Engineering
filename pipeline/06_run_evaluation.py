from __future__ import annotations

import sys
import json
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from sklearn.model_selection import train_test_split

from src.normalize import normalize_series
from src.category_mapping import contextual_category
from src.historical_lookup import HistoricalLookup
from src.similarity import MultilingualSemanticSimilarity, SemanticSimilarity
from src.cluster_evidence import ClusterEvidence
from src.engine import combine


DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = Path(__file__).parent / "outputs"

MAX_SANE_CATEGORY_ID = 10_000


def load_data():

    df = pd.read_csv(DATA_DIR / "hk_transactions_table.csv")

    raw_rows = len(df)

    df = df.dropna(
        subset=["id", "vch_desc", "category_id", "vch_type"]
    ).copy()

    df = df[df["category_id"] <= MAX_SANE_CATEGORY_ID]

    df["clean_description"] = normalize_series(df["vch_desc"])

    df = df[df["clean_description"] != ""]

    df["category_id"] = df["category_id"].astype(int)

    print(
        f"Usable labeled rows: {len(df):,} "
        f"(of {raw_rows:,} raw rows)"
    )

    print(
        f"Distinct ORIGINAL category_id values: "
        f"{df['category_id'].nunique():,}"
    )

    return df


def main():

    OUT_DIR.mkdir(exist_ok=True)

    df = load_data()

    train, test = train_test_split(
        df,
        test_size=0.20,
        random_state=42
    )

    print(
        f"Train: {len(train):,}   "
        f"Test: {len(test):,}"
    )

    # ---------------------------------------------------------
    # Layer 2 — Historical lookup
    # ---------------------------------------------------------

    print("\nTraining historical lookup...")

    hist = HistoricalLookup().fit(
        train["clean_description"],
        train["category_id"]
    )

    # ---------------------------------------------------------
    # Layer 3 — Semantic similarity
    # ---------------------------------------------------------

    backend = os.environ.get(
        "SEMANTIC_BACKEND",
        "tfidf"
    ).lower()

    if backend == "tfidf":
        semantic_class = SemanticSimilarity
    else:
        semantic_class = MultilingualSemanticSimilarity

    print(
        f"Semantic backend: {backend}"
    )

    sem = semantic_class(k=5).fit(
        train["clean_description"],
        train["category_id"]
    )

    # ---------------------------------------------------------
    # Layer 4 — Cluster evidence
    # ---------------------------------------------------------

    print("\nTraining cluster evidence...")

    cluster = ClusterEvidence().fit(
        str(DATA_DIR / "cluster_master.csv"),
        normalize_fn=lambda s: s
    )

    print(
        f"Strong clusters: "
        f"{cluster.n_strong_clusters}"
    )

    print(
        f"Weak clusters ignored: "
        f"{cluster.n_weak_clusters}"
    )

    # ---------------------------------------------------------
    # Predictions on TEST
    # ---------------------------------------------------------

    print("\nRunning predictions...")

    hist_pred = hist.lookup_batch(
        test["clean_description"]
    )

    sem_pred = sem.predict_batch(
        test["clean_description"]
    )

    cluster_pred = cluster.boost_batch(
        test["clean_description"]
    )

    # Contextual rules remain part of the production engine.
    contextual_pred = pd.DataFrame(
        {
            "contextual_category": [
                contextual_category(
                    description,
                    vch_type
                )
                for description, vch_type
                in zip(
                    test["clean_description"],
                    test["vch_type"]
                )
            ]
        },
        index=test.index,
    )

    # ---------------------------------------------------------
    # Layer 5 — Final engine
    # ---------------------------------------------------------

    result = combine(
        hist_pred,
        sem_pred,
        cluster_pred,
        contextual_pred
    )

    result.index = test.index

    # ORIGINAL category_id is the baseline ground truth
    result["true_category"] = test["category_id"]

    result["clean_description"] = test[
        "clean_description"
    ]

    result["vch_desc"] = test[
        "vch_desc"
    ]

    result["vch_type"] = test[
        "vch_type"
    ]

    result["vch_amount"] = test[
        "vch_amount"
    ]

    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------

    result["correct"] = (
        result["predicted_category"]
        == result["true_category"]
    )

    report = {}

    report["evaluation_target"] = "original_category_id"

    report["overall"] = {
        "test_rows": len(result),

        "coverage_attempted_pct": round(
            100
            * result["predicted_category"].notna().mean(),
            2
        ),

        "accuracy_all_rows_incl_unknown_as_wrong": round(
            100 * result["correct"].mean(),
            2
        ),
    }

    for decision in [
        "AUTO_CATEGORIZE",
        "REVIEW",
        "UNKNOWN"
    ]:

        subset = result[
            result["decision"] == decision
        ]

        report[decision] = {

            "count": len(subset),

            "pct_of_test_set": round(
                100 * len(subset) / len(result),
                2
            ),

            "accuracy": (
                round(
                    100 * subset["correct"].mean(),
                    2
                )
                if decision != "UNKNOWN"
                and len(subset) > 0
                else None
            ),
        }

    report[
        "evidence_source_breakdown"
    ] = (
        result[
            "evidence_source"
        ]
        .value_counts()
        .to_dict()
    )

    # ---------------------------------------------------------
    # Print results
    # ---------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("BASELINE EVALUATION")
    print("=" * 70)

    print(
        f"\nAUTO: "
        f"{report['AUTO_CATEGORIZE']['count']:,} "
        f"("
        f"{report['AUTO_CATEGORIZE']['accuracy']:.2f}% accuracy)"
    )

    print(
        f"REVIEW: "
        f"{report['REVIEW']['count']:,} "
        f"("
        f"{report['REVIEW']['accuracy']:.2f}% accuracy)"
    )

    print(
        f"UNKNOWN: "
        f"{report['UNKNOWN']['count']:,} "
        f"("
        f"{report['UNKNOWN']['pct_of_test_set']:.2f}% of test)"
    )

    print(
        f"OVERALL ACCURACY: "
        f"{report['overall']['accuracy_all_rows_incl_unknown_as_wrong']:.2f}%"
    )

    print("=" * 70)

    # ---------------------------------------------------------
    # Save report
    # ---------------------------------------------------------

    with open(
        OUT_DIR / "evaluation_report.json",
        "w"
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
            default=str
        )

    # ---------------------------------------------------------
    # Save detailed predictions
    # ---------------------------------------------------------

    cols = [
        "vch_desc",
        "clean_description",
        "vch_type",
        "vch_amount",

        "true_category",
        "predicted_category",

        "confidence",
        "decision",
        "evidence_source",
        "correct",

        "hist_category",
        "hist_purity",
        "hist_support",

        "sem_category",
        "sem_confidence",
        "sem_top_similarity",

        "cluster_match_id",

        "contextual_category",
    ]

    result[cols].to_csv(
        OUT_DIR / "test_predictions_detail.csv",
        index=False
    )

    print(
        f"\nWrote:"
        f"\n  {OUT_DIR / 'evaluation_report.json'}"
        f"\n  {OUT_DIR / 'test_predictions_detail.csv'}"
    )


if __name__ == "__main__":
    main()
