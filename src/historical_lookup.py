from __future__ import annotations
import pandas as pd


class HistoricalLookup:
    def __init__(self):
        self.table: pd.DataFrame | None = None  # normalized_description -> category, purity, support

    def fit(self, descriptions: pd.Series, categories: pd.Series) -> "HistoricalLookup":
        df = pd.DataFrame({"description": descriptions, "category": categories})
        df = df[df["description"] != ""]

        grouped = df.groupby(["description", "category"]).size().rename("count").reset_index()
        totals = grouped.groupby("description")["count"].sum().rename("support")
        top = grouped.sort_values("count", ascending=False).drop_duplicates("description")
        top = top.merge(totals, on="description")
        top["purity"] = top["count"] / top["support"]

        self.table = top.set_index("description")[["category", "purity", "support"]]
        return self

    def lookup(self, description: str) -> tuple[object, float, int] | tuple[None, float, int]:
        """Returns (category, purity, support) or (None, 0.0, 0) if unseen."""
        if self.table is None or description not in self.table.index:
            return None, 0.0, 0
        row = self.table.loc[description]
        return row["category"], float(row["purity"]), int(row["support"])

    def lookup_batch(self, descriptions: pd.Series) -> pd.DataFrame:
        results = descriptions.apply(self.lookup)
        return pd.DataFrame(results.tolist(), columns=["hist_category", "hist_purity", "hist_support"], index=descriptions.index)
