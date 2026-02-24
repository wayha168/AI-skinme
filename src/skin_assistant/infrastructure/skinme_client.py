"""
Fetch product data from SkinMe API (JSON). Saves to CSV.
For HTML scraping use bs4 in a separate scraper; this client uses requests for the REST API.
"""
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from skin_assistant.config import get_settings


SKINME_PRODUCTS_ALL = "https://backend.skinme.store/api/v1/products/all"
SKINME_BASE = "https://backend.skinme.store"


def _normalize_product(p: dict) -> dict:
    """Flatten one API product into a CSV row."""
    cat = p.get("category") or {}
    images = p.get("images") or []
    first_img = images[0] if images else {}
    download_url = first_img.get("downloadUrl") or ""
    full_image_url = (SKINME_BASE + download_url) if download_url.startswith("/") else download_url
    return {
        "id": p.get("id"),
        "name": p.get("name"),
        "brand": p.get("brand"),
        "price": p.get("price"),
        "productType": p.get("productType"),
        "inventory": p.get("inventory"),
        "description": p.get("description"),
        "howToUse": p.get("howToUse"),
        "category_id": cat.get("id"),
        "category_name": cat.get("name"),
        "image_url": full_image_url,
        "image_id": first_img.get("imageId"),
        "image_filename": first_img.get("fileName"),
        "all_image_urls": "|".join(
            (SKINME_BASE + (img.get("downloadUrl") or "")) if (img.get("downloadUrl") or "").startswith("/")
            else (img.get("downloadUrl") or "")
            for img in images
        ),
    }


def fetch_products(api_url: Optional[str] = None, timeout: int = 30) -> list[dict]:
    """
    Fetch all products from SkinMe API. Returns list of normalized product dicts.
    Uses requests (JSON API); use bs4 only when scraping HTML pages.
    """
    url = api_url or get_settings().skinme_api_url
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    if data.get("message") != "success" or "data" not in data:
        raise ValueError("Unexpected API response: missing data")
    return [_normalize_product(p) for p in data["data"]]


def products_to_csv(products: list[dict], csv_path: Path) -> None:
    """Write products to CSV."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(products)
    df.to_csv(csv_path, index=False, encoding="utf-8")


def load_existing_csv(csv_path: Path) -> pd.DataFrame:
    """Load existing CSV if present; empty DataFrame otherwise."""
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def sync_products_to_csv(
    csv_path: Optional[Path] = None,
    api_url: Optional[str] = None,
) -> dict:
    """
    Fetch current products from API and overwrite CSV with latest data.
    Returns stats: added, removed, total.
    """
    settings = get_settings()
    csv_path = csv_path or settings.skinme_products_path
    products = fetch_products(api_url=api_url)
    old_df = load_existing_csv(csv_path)
    old_ids = set(old_df["id"].astype(str).unique()) if not old_df.empty and "id" in old_df.columns else set()
    new_ids = {str(p["id"]) for p in products}
    added = len(new_ids - old_ids)
    removed = len(old_ids - new_ids)
    products_to_csv(products, csv_path)
    return {"total": len(products), "added": added, "removed": removed, "path": str(csv_path)}
