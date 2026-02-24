"""
Train product models: text (productType from name+description) and/or image (productType from image).
Requires synced data: python -m scripts.sync_products first.
Usage:
  python -m scripts.train_products                    # text only
  python -m scripts.train_products --image             # text + image
  python -m scripts.train_products --image --epochs 10
"""
import argparse
import json
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
if str(_root / "src") not in sys.path:
    sys.path.insert(0, str(_root / "src"))

from skin_assistant.models.product_trainer import (
    train_product_type_from_text,
    train_image_classifier,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", action="store_true", help="Also train image classifier (needs torch, images)")
    parser.add_argument("--label", type=str, default="productType", help="Label column: productType or category_name")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs for image model")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    results = {}
    text_result = train_product_type_from_text(label_column=args.label)
    results["text_model"] = text_result
    if "error" in text_result:
        print("Text training:", text_result["error"])
    else:
        print("Text model:", text_result.get("model_path"), "accuracy:", text_result.get("accuracy"))

    if args.image:
        img_result = train_image_classifier(
            label_column=args.label,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        results["image_model"] = img_result
        if "error" in img_result:
            print("Image training:", img_result["error"])
        else:
            print("Image model:", img_result.get("model_path"))

    print(json.dumps(results, indent=2))
    return 0 if "error" not in text_result else 1


if __name__ == "__main__":
    sys.exit(main())
