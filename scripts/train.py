"""
Train the Skin Assistant intent classification model.
Usage:
  python -m scripts.train
  python -m scripts.train --data data/intent_training.csv --output models/artifacts
"""
import argparse
import json
import sys
from pathlib import Path

# Ensure src is on path when run as script
_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
if str(_root / "src") not in sys.path:
    sys.path.insert(0, str(_root / "src"))

from skin_assistant.models.trainer import train


def main():
    parser = argparse.ArgumentParser(description="Train intent classification model")
    parser.add_argument("--data", type=Path, default=None, help="CSV with columns: text, intent")
    parser.add_argument("--output", type=Path, default=None, help="Directory to save model artifact")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    metrics = train(
        data_path=args.data,
        output_dir=args.output,
        test_size=args.test_size,
        random_state=args.seed,
    )
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
