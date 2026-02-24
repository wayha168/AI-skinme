"""Application settings and paths."""
import os
from pathlib import Path


def get_settings():
    """Return settings object (paths + optional backend URL for DB integration)."""
    root = Path(__file__).resolve().parents[3]  # config -> skin_assistant -> src -> project root
    # Spring (or other) backend URL for saving data to database; e.g. http://localhost:8080
    backend_url = os.environ.get("SPRING_BACKEND_URL") or os.environ.get("BACKEND_WEBHOOK_URL") or ""
    return type("Settings", (), {
        "project_root": root,
        "data_dir": root / "data",
        "ingredients_path": root / "data" / "ingredientsList.csv",
        "products_path": root / "data" / "skincare_products_clean.csv",
        "models_dir": root / "models" / "artifacts",
        # SkinMe API & scraped product data
        "skinme_api_url": "https://backend.skinme.store/api/v1/products/all",
        "skinme_base_url": "https://backend.skinme.store",
        "skinme_products_path": root / "data" / "skinme_products.csv",
        "product_images_dir": root / "data" / "product_images",
        # Skin disease / condition images for training (folder or CSV with image_name, condition)
        "skin_disease_images_dir": root / "data" / "skin_disease_images",
        "skin_disease_labels_path": root / "data" / "skin_disease_labels.csv",
        # Backend integration: Spring URL to forward save requests (persist to DB)
        "backend_url": backend_url.rstrip("/") if backend_url else "",
    })()
