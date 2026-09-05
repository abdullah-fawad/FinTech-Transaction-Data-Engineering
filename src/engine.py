from __future__ import annotations
import pandas as pd

AUTO_THRESHOLD = 0.90
REVIEW_THRESHOLD = 0.70

# supports the "more repeats = more trust" idea without letting support alone
# override purity — a description seen 3 times at 100% purity should already
# be trustworthy; supports beyond ~5 add only marginal extra confidence.
SUPPORT_SATURATION = 5

# when the historical and semantic layers independently agree on the same
# category, that agreement is itself evidence — two different methods
# landing on the same answer is more trustworthy than either alone.
AGREEMENT_BONUS = 0.15


def _support_factor(support: int) -> float:
    return min(support, SUPPORT_SATURATION) / SUPPORT_SATURATION


def combine(
    hist_df: pd.DataFrame,
    sem_df: pd.DataFrame,
    cluster_df: pd.DataFrame,
    contextual_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    df = pd.concat([hist_df, sem_df, cluster_df], axis=1)
    if contextual_df is not None:
        df = pd.concat([df, contextual_df], axis=1)

    final_category = []
    final_confidence = []
    source = []

    for _, row in df.iterrows():
        contextual_category = row.get("contextual_category")
        has_contextual = pd.notna(contextual_category)
        has_hist = pd.notna(row["hist_category"]) and row["hist_support"] >= 1
        has_sem = pd.notna(row["sem_category"])

        hist_conf = row["hist_purity"] * _support_factor(row["hist_support"]) if has_hist else 0.0
        sem_conf = row["sem_confidence"] if has_sem else 0.0

        if has_contextual:
            cat, conf, src = contextual_category, 0.99, "contextual_rule"
        elif has_hist and has_sem and row["hist_category"] == row["sem_category"]:
            cat = row["hist_category"]
            conf = min(max(hist_conf, sem_conf) + AGREEMENT_BONUS, 0.99)
            src = "historical+semantic_agree"
        elif has_hist and hist_conf >= sem_conf:
            cat, conf, src = row["hist_category"], hist_conf, "historical"
        elif has_sem:
            cat, conf, src = row["sem_category"], sem_conf, "semantic"
        else:
            cat, conf, src = None, 0.0, "none"

        if cat is not None and row["cluster_boost"] > 0:
            conf = min(conf + row["cluster_boost"], 0.99)
            src += "+cluster"

        final_category.append(cat)
        final_confidence.append(round(float(conf), 4))
        source.append(src)

    df["predicted_category"] = final_category
    df["confidence"] = final_confidence
    df["evidence_source"] = source
    df["decision"] = pd.cut(
        df["confidence"],
        bins=[-0.01, REVIEW_THRESHOLD, AUTO_THRESHOLD, 1.01],
        labels=["UNKNOWN", "REVIEW", "AUTO_CATEGORIZE"],
    )
    # transactions with no category candidate at all are always UNKNOWN regardless of score
    df.loc[df["predicted_category"].isna(), "decision"] = "UNKNOWN"

    output_columns = ["predicted_category", "confidence", "evidence_source", "decision",
                      "hist_category", "hist_purity", "hist_support",
                      "sem_category", "sem_confidence", "sem_top_similarity", "cluster_match_id"]
    if "contextual_category" in df:
        output_columns.append("contextual_category")
    return df[output_columns]
