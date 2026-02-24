"""
Skin care knowledge base: loads ingredients and products from CSV and supports search.
"""
from pathlib import Path
import pandas as pd
import re


def _data_dir():
    return Path(__file__).resolve().parent / "data"


def load_ingredients():
    """Load ingredients list CSV. Returns DataFrame with text search field."""
    path = _data_dir() / "ingredientsList.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    # Build searchable text (lowercase for matching)
    cols = ["name", "short_description", "what_is_it", "what_does_it_do", "who_is_it_good_for", "who_should_avoid"]
    existing = [c for c in cols if c in df.columns]
    df["_search_text"] = df[existing].fillna("").astype(str).agg(" ".join, axis=1).str.lower()
    return df


def load_products():
    """Load skincare products CSV (has clean_ingreds for ingredient matching)."""
    path = _data_dir() / "skincare_products_clean.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "clean_ingreds" in df.columns:
        df["_ingreds_str"] = df["clean_ingreds"].fillna("").astype(str).str.lower()
    return df


def load_skinme_products():
    """Load SkinMe products CSV (from sync). Used for concern-based recommendations."""
    path = _data_dir() / "skinme_products.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    # Build searchable text: name, description, productType, category_name
    parts = [
        df.get("name", pd.Series("")).fillna(""),
        df.get("description", pd.Series("")).fillna(""),
        df.get("productType", pd.Series("")).fillna("").astype(str),
        df.get("category_name", pd.Series("")).fillna("").astype(str),
    ]
    df["_search_text"] = pd.concat(parts, axis=1).agg(" ".join, axis=1).str.lower()
    return df


def search_skinme_by_concern(
    concern: str,
    skinme_df: pd.DataFrame,
    product_type: str | None = None,
    top_k: int = 5,
):
    """Find SkinMe products matching a concern (name, description, productType, category)."""
    if skinme_df.empty or not concern.strip():
        return []
    c = concern.strip().lower()
    subset = skinme_df.copy()
    if product_type:
        subset = subset[
            subset["productType"]
            .fillna("")
            .astype(str)
            .str.lower()
            .str.contains(product_type.lower(), na=False)
        ]
    mask = subset["_search_text"].str.contains(re.escape(c), na=False)
    # Also match individual tokens for multi-word concerns
    if not mask.any():
        for t in set(re.findall(r"\w+", c)):
            if len(t) >= 3:
                mask = mask | subset["_search_text"].str.contains(re.escape(t), na=False)
    subset = subset[mask]
    # Normalize to same shape as load_products() output for _format_product
    out = []
    for _, row in subset.head(top_k).iterrows():
        out.append({
            "product_name": row.get("name"),
            "product_type": row.get("productType"),
            "price": str(row.get("price", "")),
            "product_url": row.get("image_url") or "",
        })
    return out


def search_ingredients(query: str, ingredients_df: pd.DataFrame, top_k: int = 5):
    """Find ingredients matching query (keyword match in name and descriptions)."""
    if ingredients_df.empty or not query.strip():
        return []
    q = query.strip().lower()
    # Prefer name match, then text match
    name_match = ingredients_df["name"].fillna("").astype(str).str.lower().str.contains(re.escape(q), na=False)
    text_match = ingredients_df["_search_text"].str.contains(re.escape(q), na=False)
    mask = name_match | text_match
    if not mask.any():
        # Try token match
        tokens = set(re.findall(r"\w+", q))
        for t in tokens:
            if len(t) < 3:
                continue
            mask = mask | ingredients_df["_search_text"].str.contains(re.escape(t), na=False)
    out = ingredients_df[mask].head(top_k)
    return out.to_dict("records")


def search_products_by_concern(concern: str, products_df: pd.DataFrame, product_type: str | None = None, top_k: int = 5):
    """Find products that might help a concern (by ingredient text or product type)."""
    if products_df.empty or not concern.strip():
        return []
    c = concern.strip().lower()
    subset = products_df
    if product_type:
        subset = subset[subset["product_type"].fillna("").astype(str).str.lower().str.contains(product_type.lower(), na=False)]
    if "_ingreds_str" in subset.columns:
        mask = subset["_ingreds_str"].str.contains(re.escape(c), na=False)
        # Also match product name
        name_match = subset["product_name"].fillna("").astype(str).str.lower().str.contains(re.escape(c), na=False)
        mask = mask | name_match
        subset = subset[mask]
    return subset.head(top_k).to_dict("records")


def get_ingredient_by_name(name: str, ingredients_df: pd.DataFrame):
    """Get a single ingredient by exact or fuzzy name."""
    if ingredients_df.empty or not name.strip():
        return None
    n = name.strip().lower()
    for _, row in ingredients_df.iterrows():
        if row.get("name") and str(row["name"]).lower() == n:
            return row.to_dict()
    # fuzzy
    name_match = ingredients_df["name"].fillna("").astype(str).str.lower().str.contains(re.escape(n), na=False)
    if name_match.any():
        return ingredients_df[name_match].iloc[0].to_dict()
    return None


def get_products_containing_ingredient(ingredient_name: str, products_df: pd.DataFrame, top_k: int = 5):
    """Get products that contain a given ingredient."""
    if products_df.empty or "clean_ingreds" not in products_df.columns or not ingredient_name.strip():
        return []
    key = ingredient_name.strip().lower()
    mask = products_df["_ingreds_str"].str.contains(re.escape(key), na=False)
    return products_df[mask].head(top_k).to_dict("records")
