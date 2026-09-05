from __future__ import annotations

from pathlib import Path

import pandas as pd


MAPPING_PATH = Path(__file__).parents[1] / "data" / "category_mapping_approved.csv"
CONTEXTUAL_MAPPING_PATH = Path(__file__).parents[1] / "data" / "contextual_rules_approved.csv"


def _load_mapping() -> dict[tuple[str, int], int]:
    if not MAPPING_PATH.exists():
        return {}
    mapping = pd.read_csv(MAPPING_PATH)
    mapping = mapping[mapping["approved"].astype(str).str.lower().eq("true")]
    return {
        (row.normalized_description, int(row.source_category_id)): int(row.canonical_category_id)
        for row in mapping.itertuples()
    }


APPROVED_CATEGORY_MAP = _load_mapping()


def _load_contextual_mapping() -> dict[tuple[str, str], int]:
    if not CONTEXTUAL_MAPPING_PATH.exists():
        return {}
    mapping = pd.read_csv(CONTEXTUAL_MAPPING_PATH)
    mapping = mapping[mapping["approved"].astype(str).str.lower().eq("true")]
    return {
        (row.normalized_description, row.vch_type): int(row.canonical_category_id)
        for row in mapping.itertuples()
    }


APPROVED_CONTEXTUAL_MAP = _load_contextual_mapping()


def canonical_category(description: str, category_id: int, vch_type: str | None = None) -> int:
    """Return the approved canonical ID for one labeled row."""
    category = APPROVED_CATEGORY_MAP.get((description, int(category_id)), int(category_id))
    if vch_type is not None:
        category = APPROVED_CONTEXTUAL_MAP.get((description, vch_type), category)
    return category


def contextual_category(description: str, vch_type: str) -> int | None:
    return APPROVED_CONTEXTUAL_MAP.get((description, vch_type))