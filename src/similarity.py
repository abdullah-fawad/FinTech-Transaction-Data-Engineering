from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


def _build_reference(descriptions: pd.Series, categories: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"description": descriptions, "category": categories})
    df = df[df["description"] != ""]
    grouped = df.groupby(["description", "category"]).size().rename("count").reset_index()
    totals = grouped.groupby("description")["count"].sum().rename("support")
    top = grouped.sort_values("count", ascending=False).drop_duplicates("description")
    top = top.merge(totals, on="description")
    top["purity"] = top["count"] / top["support"]
    return top


class SemanticSimilarity:
    def __init__(self, k: int = 5):
        self.k = k
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
        self.nn: NearestNeighbors | None = None
        self.ref_categories: np.ndarray | None = None
        self.ref_purity: np.ndarray | None = None

    def fit(self, descriptions: pd.Series, categories: pd.Series) -> "SemanticSimilarity":
        """Builds the reference index from UNIQUE normalized descriptions
        (each with its dominant historical category + purity), not every
        raw row — this keeps the index small and avoids a single common
        description dominating the neighbor vote just by sheer row count."""
        top = _build_reference(descriptions, categories)

        self.ref_descriptions = top["description"].tolist()
        self.ref_categories = top["category"].values
        self.ref_purity = top["purity"].values

        X = self.vectorizer.fit_transform(self.ref_descriptions)
        self.nn = NearestNeighbors(n_neighbors=min(self.k, len(self.ref_descriptions)), metric="cosine")
        self.nn.fit(X)
        return self

    def predict_batch(self, descriptions: pd.Series) -> pd.DataFrame:
        desc_list = descriptions.fillna("").tolist()
        non_empty_idx = [i for i, d in enumerate(desc_list) if d.strip()]
        results = [(None, 0.0, 0.0)] * len(desc_list)

        if non_empty_idx:
            X = self.vectorizer.transform([desc_list[i] for i in non_empty_idx])
            distances, indices = self.nn.kneighbors(X)
            similarities = 1 - distances  # cosine distance -> similarity

            for row, orig_idx in enumerate(non_empty_idx):
                neighbor_idxs = indices[row]
                neighbor_sims = similarities[row]
                neighbor_cats = self.ref_categories[neighbor_idxs]
                neighbor_purities = self.ref_purity[neighbor_idxs]

                # weighted vote: weight = similarity * that neighbor's own historical purity
                weights = neighbor_sims * neighbor_purities
                if weights.sum() <= 0:
                    results[orig_idx] = (None, 0.0, 0.0)
                    continue

                vote = {}
                for cat, w in zip(neighbor_cats, weights):
                    vote[cat] = vote.get(cat, 0.0) + w
                best_cat = max(vote, key=vote.get)
                # confidence = top similarity to a same-category neighbor, scaled by vote concentration
                top_sim = neighbor_sims[neighbor_cats == best_cat].max()
                concentration = vote[best_cat] / weights.sum()
                confidence = float(top_sim * concentration)
                results[orig_idx] = (best_cat, confidence, float(top_sim))

        return pd.DataFrame(results, columns=["sem_category", "sem_confidence", "sem_top_similarity"], index=descriptions.index)


class MultilingualSemanticSimilarity:
    """Sentence-transformer replacement with the same prediction contract."""

    def __init__(self, k: int = 5, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        from sentence_transformers import SentenceTransformer

        self.k = k
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.nn: NearestNeighbors | None = None
        self.ref_categories: np.ndarray | None = None
        self.ref_purity: np.ndarray | None = None

    def fit(self, descriptions: pd.Series, categories: pd.Series) -> "MultilingualSemanticSimilarity":
        top = _build_reference(descriptions, categories)
        self.ref_descriptions = top["description"].tolist()
        self.ref_categories = top["category"].values
        self.ref_purity = top["purity"].values
        embeddings = self.model.encode(
            self.ref_descriptions,
            batch_size=128,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        self.nn = NearestNeighbors(
            n_neighbors=min(self.k, len(self.ref_descriptions)), metric="cosine"
        ).fit(embeddings)
        return self

    def predict_batch(self, descriptions: pd.Series) -> pd.DataFrame:
        desc_list = descriptions.fillna("").tolist()
        results = [(None, 0.0, 0.0)] * len(desc_list)
        non_empty_idx = [i for i, description in enumerate(desc_list) if description.strip()]
        if non_empty_idx:
            embeddings = self.model.encode(
                [desc_list[i] for i in non_empty_idx],
                batch_size=128,
                show_progress_bar=True,
                normalize_embeddings=True,
            )
            distances, indices = self.nn.kneighbors(embeddings)
            similarities = 1 - distances
            for row, original_index in enumerate(non_empty_idx):
                neighbor_indices = indices[row]
                neighbor_similarities = similarities[row]
                neighbor_categories = self.ref_categories[neighbor_indices]
                neighbor_purities = self.ref_purity[neighbor_indices]
                weights = neighbor_similarities * neighbor_purities
                if weights.sum() <= 0:
                    continue
                votes: dict[object, float] = {}
                for category, weight in zip(neighbor_categories, weights):
                    votes[category] = votes.get(category, 0.0) + weight
                best_category = max(votes, key=votes.get)
                top_similarity = neighbor_similarities[neighbor_categories == best_category].max()
                concentration = votes[best_category] / weights.sum()
                results[original_index] = (
                    best_category,
                    float(top_similarity * concentration),
                    float(top_similarity),
                )
        return pd.DataFrame(
            results,
            columns=["sem_category", "sem_confidence", "sem_top_similarity"],
            index=descriptions.index,
        )
