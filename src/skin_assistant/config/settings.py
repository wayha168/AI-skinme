"""Application settings and paths."""
import os
from pathlib import Path


def get_settings():
    """Return settings object (paths + optional backend URL and MySQL for DB integration)."""
    root = Path(__file__).resolve().parents[3]  # config -> skin_assistant -> src -> project root
    # Spring (or other) backend URL for saving data to database; e.g. http://localhost:8080
    backend_url = os.environ.get("SPRING_BACKEND_URL") or os.environ.get("BACKEND_WEBHOOK_URL") or ""
    # Optional: MySQL (skinme_db) for product lookups when "use database" is enabled in chat
    mysql_host = os.environ.get("MYSQL_HOST", "").strip()
    mysql_port = int(os.environ.get("MYSQL_PORT", "3306"))
    mysql_user = os.environ.get("MYSQL_USER", "").strip()
    mysql_password = os.environ.get("MYSQL_PASSWORD", "").strip()
    mysql_database = os.environ.get("MYSQL_DATABASE", "").strip()
    mysql_products_table = os.environ.get("MYSQL_PRODUCTS_TABLE", "product").strip() or "product"
    use_mysql_db = bool(mysql_host and mysql_user and mysql_database)
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
        # Optional MySQL (skinme_db) for chat "check with database" option
        "mysql_host": mysql_host,
        "mysql_port": mysql_port,
        "mysql_user": mysql_user,
        "mysql_password": mysql_password,
        "mysql_database": mysql_database,
        "mysql_products_table": mysql_products_table,
        "use_mysql_db": use_mysql_db,
    })()
