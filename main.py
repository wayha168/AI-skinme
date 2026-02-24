"""
Skin Assistant — entry point.
  python main.py              # run API
  python main.py sync         # fetch SkinMe API -> CSV, download images, cleanup
  python main.py train        # train intent model
  python main.py train-products [--image]  # train product text (+ image) models
  python main.py train-skin-condition      # train skin condition classifier from images (for recommendations)
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


def run_api(port: int = 8000, host: str = "0.0.0.0") -> None:
    import uvicorn
    from skin_assistant.api.app import app
    uvicorn.run(app, host=host, port=port)


def run_train() -> int:
    from skin_assistant.models.trainer import train
    metrics = train()
    print("Training done:", metrics)
    return 0


def run_sync(no_download: bool = False, no_cleanup: bool = False) -> int:
    from scripts.sync_products import do_sync
    return do_sync(no_download=no_download, no_cleanup=no_cleanup, no_sync=False)


def run_train_products(image: bool = False, **kwargs) -> int:
    from skin_assistant.models.product_trainer import train_product_type_from_text, train_image_classifier
    from skin_assistant.config import get_settings
    label = kwargs.get("label_column", "productType")
    r1 = train_product_type_from_text(label_column=label)
    print("Text model:", r1)
    if r1.get("error"):
        return 1
    if image:
        r2 = train_image_classifier(label_column=label, **{k: v for k, v in kwargs.items() if k in ("epochs", "batch_size")})
        print("Image model:", r2)
    return 0


def run_train_skin_condition(**kwargs) -> int:
    from skin_assistant.models.skin_condition_trainer import train_skin_condition_classifier
    result = train_skin_condition_classifier(**kwargs)
    print("Skin condition model:", result)
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
    parser.add_argument("--no-download", action="store_true", help="sync: skip image download")
    parser.add_argument("--no-cleanup", action="store_true", help="sync: skip deleting unused images")
    parser.add_argument("--image", action="store_true", help="train-products: also train image model")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    if args.command == "serve":
        run_api(port=args.port, host=args.host)
        return 0
    if args.command == "sync":
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
