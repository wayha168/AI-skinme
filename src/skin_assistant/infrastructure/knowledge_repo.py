"""Data layer: load and search ingredients and products from CSV and optional MySQL (skinme_db)."""
import re
from pathlib import Path
from typing import Optional

import pandas as pd

from skin_assistant.config import get_settings


def _get_db_client():
    """Lazy singleton for optional MySQL client."""
    try:
        from skin_assistant.infrastructure.skinme_db import SkinMeDBClient
        c = SkinMeDBClient()
        return c if c.is_available() else None
    except Exception:
        return None


class KnowledgeRepository:
    """Repository for ingredients, skincare products, SkinMe products, and optional MySQL skinme_db."""

    def __init__(self, data_dir: Optional[Path] = None):
        settings = get_settings()
        self._data_dir = data_dir or settings.data_dir
        self._ingredients_df: Optional[pd.DataFrame] = None
        self._products_df: Optional[pd.DataFrame] = None
        self._skinme_df: Optional[pd.DataFrame] = None
        self._db_client = None

    def _ensure_ingredients(self) -> pd.DataFrame:
        if self._ingredients_df is None:
            path = self._data_dir / "ingredientsList.csv"
            if not path.exists():
                self._ingredients_df = pd.DataFrame()
            else:
                df = pd.read_csv(path)
                cols = ["name", "short_description", "what_is_it", "what_does_it_do", "who_is_it_good_for", "who_should_avoid"]
                existing = [c for c in cols if c in df.columns]
                df["_search_text"] = df[existing].fillna("").astype(str).agg(" ".join, axis=1).str.lower()
                self._ingredients_df = df
        return self._ingredients_df

    def _ensure_products(self) -> pd.DataFrame:
        if self._products_df is None:
            path = self._data_dir / "skincare_products_clean.csv"
            if not path.exists():
                self._products_df = pd.DataFrame()
            else:
                df = pd.read_csv(path)
                if "clean_ingreds" in df.columns:
                    df["_ingreds_str"] = df["clean_ingreds"].fillna("").astype(str).str.lower()
                self._products_df = df
        return self._products_df

    def _ensure_skinme(self) -> pd.DataFrame:
        if self._skinme_df is None:
            path = get_settings().skinme_products_path
            if not path.exists():
                self._skinme_df = pd.DataFrame()
            else:
                df = pd.read_csv(path)
                parts = [
                    df.get("name", pd.Series("")).fillna(""),
                    df.get("description", pd.Series("")).fillna(""),
                    df.get("productType", pd.Series("")).fillna("").astype(str),
                    df.get("category_name", pd.Series("")).fillna("").astype(str),
                ]
                df["_search_text"] = pd.concat(parts, axis=1).agg(" ".join, axis=1).str.lower()
                self._skinme_df = df
        return self._skinme_df

    def _search_skinme_by_concern(
        self, concern: str, product_type: Optional[str] = None, top_k: int = 5
    ) -> list[dict]:
        df = self._ensure_skinme()
        if df.empty or not concern or not concern.strip():
            return []
        c = concern.strip().lower()
        subset = df
        if product_type:
            subset = subset[
                subset["productType"].fillna("").astype(str).str.lower().str.contains(product_type.lower(), na=False)
            ]
        mask = subset["_search_text"].str.contains(re.escape(c), na=False)
        if not mask.any():
            for t in set(re.findall(r"\w+", c)):
                if len(t) >= 3:
                    mask = mask | subset["_search_text"].str.contains(re.escape(t), na=False)
        out = []
        for _, row in subset[mask].head(top_k).iterrows():
            out.append({
                "product_name": row.get("name"),
                "product_type": row.get("productType"),
                "price": str(row.get("price", "")),
                "product_url": row.get("image_url") or "",
            })
        return out

    def search_ingredients(self, query: str, top_k: int = 5) -> list[dict]:
        if not query or not query.strip():
            return []
        df = self._ensure_ingredients()
        if df.empty:
            return []
        q = query.strip().lower()
        name_match = df["name"].fillna("").astype(str).str.lower().str.contains(re.escape(q), na=False)
        text_match = df["_search_text"].str.contains(re.escape(q), na=False)
        mask = name_match | text_match
        if not mask.any():
            for t in set(re.findall(r"\w+", q)):
                if len(t) >= 3:
                    mask = mask | df["_search_text"].str.contains(re.escape(t), na=False)
        return df[mask].head(top_k).to_dict("records")

    def get_ingredient_by_name(self, name: str) -> Optional[dict]:
        if not name or not name.strip():
            return None
        df = self._ensure_ingredients()
        if df.empty:
            return None
        n = name.strip().lower()
        for _, row in df.iterrows():
            if row.get("name") and str(row["name"]).lower() == n:
                return row.to_dict()
        name_match = df["name"].fillna("").astype(str).str.lower().str.contains(re.escape(n), na=False)
        if name_match.any():
            return df[name_match].iloc[0].to_dict()
        return None

    def _get_db_client(self):
        if self._db_client is None:
            self._db_client = _get_db_client()
        return self._db_client

    def search_products_by_concern(
        self, concern: str, product_type: Optional[str] = None, top_k: int = 5, use_database: bool = False
    ) -> list[dict]:
        """Search products by concern. If use_database=True and MySQL is configured, query skinme_db first."""
        if use_database:
            client = self._get_db_client()
            if client:
                db_hits = client.search_products_by_concern(concern, product_type=product_type, top_k=top_k)
                if db_hits:
                    return db_hits
        skinme_hits = self._search_skinme_by_concern(concern, product_type=product_type, top_k=top_k)
        if skinme_hits:
            return skinme_hits
        if not concern or not concern.strip():
            return []
        df = self._ensure_products()
        if df.empty:
            return []
        c = concern.strip().lower()
        subset = df
        if product_type:
            subset = subset[
                subset["product_type"].fillna("").astype(str).str.lower().str.contains(product_type.lower(), na=False)
            ]
        if "_ingreds_str" in subset.columns:
            mask = subset["_ingreds_str"].str.contains(re.escape(c), na=False)
            name_match = subset["product_name"].fillna("").astype(str).str.lower().str.contains(re.escape(c), na=False)
            mask = mask | name_match
            subset = subset[mask]
        return subset.head(top_k).to_dict("records")

    def get_products_containing_ingredient(self, ingredient_name: str, top_k: int = 5) -> list[dict]:
        if not ingredient_name or not ingredient_name.strip():
            return []
        df = self._ensure_products()
        if df.empty or "clean_ingreds" not in df.columns:
            return []
        key = ingredient_name.strip().lower()
        mask = df["_ingreds_str"].str.contains(re.escape(key), na=False)
        return df[mask].head(top_k).to_dict("records")
