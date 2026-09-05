from __future__ import annotations
import re
import pandas as pd

MIN_CLUSTER_PURITY = 0.60   
MIN_CLUSTER_SIZE = 20        
BOOST_AMOUNT = 0.07          

def _parse_top_descriptions(raw: str) -> list[tuple[str, int]]:
    pairs = []
    if not isinstance(raw, str):
        return pairs
    for item in raw.split(" | "):
        item = item.strip()
        if not item or "(" not in item:
            continue
        desc, count = item.rsplit(" (", 1)
        try:
            pairs.append((desc.strip(), int(count.rstrip(")"))))
        except ValueError:
            continue
    return pairs


class ClusterEvidence:
    def __init__(self):
        self.strong_terms: dict[str, dict] = {}  # normalized term -> {cluster_id, purity, size}

    def fit(self, cluster_master_path: str, normalize_fn) -> "ClusterEvidence":
        df = pd.read_csv(cluster_master_path)
        df = df[df["discovered_cluster"] != -1]

        n_strong, n_weak = 0, 0
        for _, row in df.iterrows():
            pairs = _parse_top_descriptions(row["top_descriptions"])
            if not pairs:
                continue
            shown_total = sum(c for _, c in pairs)
            top1_desc, top1_count = pairs[0]
            purity_proxy = top1_count / shown_total if shown_total else 0

            if row["cluster_size"] < MIN_CLUSTER_SIZE or purity_proxy < MIN_CLUSTER_PURITY:
                n_weak += 1
                continue
            n_strong += 1

            # register the dominant term (and its normalized form) as strong evidence
            norm_term = normalize_fn(top1_desc)
            if norm_term:
                self.strong_terms[norm_term] = {
                    "cluster_id": int(row["discovered_cluster"]),
                    "purity_proxy": round(purity_proxy, 3),
                    "cluster_size": int(row["cluster_size"]),
                }

        self.n_strong_clusters = n_strong
        self.n_weak_clusters = n_weak
        return self

    def boost_for(self, normalized_description: str) -> tuple[float, dict | None]:
        """Returns (boost_amount, evidence_dict_or_None). Matches on whole-term
        containment: the cluster's dominant term appearing as a token run
        inside the (already-normalized) description."""
        if not normalized_description:
            return 0.0, None
        for term, evidence in self.strong_terms.items():
            if term and re.search(rf"\b{re.escape(term)}\b", normalized_description):
                return BOOST_AMOUNT, evidence
        return 0.0, None

    def boost_batch(self, descriptions: pd.Series) -> pd.DataFrame:
        results = descriptions.apply(self.boost_for)
        boosts = results.apply(lambda t: t[0])
        cluster_ids = results.apply(lambda t: t[1]["cluster_id"] if t[1] else None)
        return pd.DataFrame({"cluster_boost": boosts, "cluster_match_id": cluster_ids}, index=descriptions.index)
