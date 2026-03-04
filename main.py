"""
Skin Assistant — entry point.
  python main.py                    # run API
  python main.py --sync-first       # sync product data from SkinMe API, then run API
  python main.py sync               # fetch SkinMe API -> CSV, download images, cleanup
  python main.py train              # train intent model
  python main.py train-products [--image]  # train product text (+ image) models
  python main.py train-skin-condition       # train skin condition classifier from images (for recommendations)

  Product data: sync pulls from SkinMe API into data/skinme_products.csv. Optional MySQL (skinme_db)
  is used at runtime when "Check with database" is enabled (set MYSQL_* in .env).
"""
import argparse
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent
_src = _root / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

try:
    from console_style import (
        print_header,
        print_section,
        print_success,
        print_info,
        print_warning,
        print_error,
        print_key_value,
        print_line,
    )
except ImportError:
    def print_header(app_name="Skinme AI", tagline=""):
        print("\n  Skinme AI - Skin Assistant\n")
    def print_section(title, char="-"):
        print(f"  {title}\n  {char * 50}")
    def print_success(msg): print("  [OK]", msg)
    def print_info(msg): print("  (i)", msg)
    def print_warning(msg): print("  (!)", msg)
    def print_error(msg): print("  (x)", msg)
    def print_key_value(k, v, indent=2): print(" " * indent + f"{k}: {v}")
    def print_line(char="-", length=50): print("  " + char * length)


def run_api(port: int = 8000, host: str = "0.0.0.0") -> None:
    import uvicorn
    from skin_assistant.api.app import app
    print_section("API Server", "-")
    print_key_value("Host", host)
    print_key_value("Port", str(port))
    print_key_value("Docs", f"http://127.0.0.1:{port}/docs")
    print_line()
    uvicorn.run(app, host=host, port=port)


def run_train() -> int:
    from skin_assistant.models.trainer import train
    print_section("Intent model training", "-")
    metrics = train()
    print_success(f"Training done: {metrics}")
    return 0


def run_sync(no_download: bool = False, no_cleanup: bool = False) -> int:
    from scripts.sync_products import do_sync
    return do_sync(no_download=no_download, no_cleanup=no_cleanup, no_sync=False)


def run_train_products(image: bool = False, **kwargs) -> int:
    from skin_assistant.models.product_trainer import train_product_type_from_text, train_image_classifier
    from skin_assistant.config import get_settings
    print_section("Product models training", "-")
    label = kwargs.get("label_column", "productType")
    r1 = train_product_type_from_text(label_column=label)
    print_key_value("Text model", str(r1))
    if r1.get("error"):
        print_error(r1.get("error", "Training failed"))
        return 1
    if image:
        r2 = train_image_classifier(label_column=label, **{k: v for k, v in kwargs.items() if k in ("epochs", "batch_size")})
        print_key_value("Image model", str(r2))
    print_success("Product training complete")
    return 0


def run_train_skin_condition(**kwargs) -> int:
    from skin_assistant.models.skin_condition_trainer import train_skin_condition_classifier
    print_section("Skin condition classifier", "-")
    result = train_skin_condition_classifier(**kwargs)
    if result.get("error"):
        print_error(result.get("error", "Training failed"))
    else:
        print_key_value("Result", str(result))
        print_success("Skin condition model trained")
    return 1 if result.get("error") else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Skin Assistant")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument(
        "command",
        nargs="?",
        default="serve",
        choices=["serve", "sync", "train", "train-products", "train-skin-condition"],
        help="serve | sync | train | train-products | train-skin-condition",
    )
    parser.add_argument(
        "--sync-first",
        action="store_true",
        help="before serve: fetch product data from SkinMe API to CSV and download images, then start API",
    )
    parser.add_argument("--no-download", action="store_true", help="sync: skip image download")
    parser.add_argument("--no-cleanup", action="store_true", help="sync: skip deleting unused images")
    parser.add_argument("--image", action="store_true", help="train-products: also train image model")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    print_header("Skinme AI", "Skin Assistant - Ingredients & Product Recommendations")
    print_section("Command", "-")
    print_key_value("Mode", "serve (API)" if args.command == "serve" else args.command)
    print_line()

    if args.command == "serve":
        if args.sync_first:
            print_section("Sync", "-")
            print_info("Syncing product data from SkinMe API...")
            if run_sync(no_download=args.no_download, no_cleanup=args.no_cleanup) != 0:
                print_warning("Sync had errors; starting API anyway.")
            print_success("Sync finished")
            print_line()
        run_api(port=args.port, host=args.host)
        return 0
    if args.command == "sync":
        print_section("Sync products", "-")
        return run_sync(no_download=args.no_download, no_cleanup=args.no_cleanup)
    if args.command == "train":
        return run_train()
    if args.command == "train-products":
        return run_train_products(
            image=args.image,
            label_column="productType",
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
    if args.command == "train-skin-condition":
        return run_train_skin_condition(epochs=args.epochs, batch_size=args.batch_size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
