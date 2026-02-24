"""
Training pipeline for intent classification model (scikit-learn).
Saves vectorizer + classifier to models/artifacts for use by the API.
"""
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from skin_assistant.config import get_settings
from skin_assistant.models.intent_labels import INTENT_LABELS, INTENT_LABEL_TO_ID


# Example training data: (text, intent). Expand with real user queries for better accuracy.
DEFAULT_TRAINING_DATA = [
    ("hi", "greeting"),
    ("hello", "greeting"),
    ("hey there", "greeting"),
    ("what is niacinamide", "ingredient_info"),
    ("tell me about hyaluronic acid", "ingredient_info"),
    ("what does retinol do", "ingredient_info"),
    ("explain vitamin c", "ingredient_info"),
    ("products with niacinamide", "products_with_ingredient"),
    ("products containing hyaluronic acid", "products_with_ingredient"),
    ("recommend products with retinol", "products_with_ingredient"),
    ("recommend for dry skin", "product_recommendation"),
    ("I have acne", "product_recommendation"),
    ("products for sensitive skin", "product_recommendation"),
    ("best moisturiser for oily skin", "product_recommendation"),
    ("ingredients for hydration", "general_ingredient_search"),
    ("search ceramide", "general_ingredient_search"),
    ("something random", "other"),
]


def build_pipeline() -> Pipeline:
    """Build TF-IDF + LogisticRegression pipeline."""
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=1)),
            ("clf", LogisticRegression(max_iter=500, C=0.5, random_state=42)),
        ]
    )


def load_training_data(csv_path: Optional[Path] = None) -> tuple[list[str], list[int]]:
    """
    Load training data. If csv_path exists, expect columns 'text' and 'intent'.
    Otherwise use DEFAULT_TRAINING_DATA.
    """
    if csv_path and csv_path.exists():
        df = pd.read_csv(csv_path)
        texts = df["text"].astype(str).tolist()
        intents = df["intent"].map(INTENT_LABEL_TO_ID).tolist()
        return texts, intents
    texts = [t for t, _ in DEFAULT_TRAINING_DATA]
    intents = [INTENT_LABEL_TO_ID[l] for _, l in DEFAULT_TRAINING_DATA]
    return texts, intents


def train(
    data_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Train intent classifier and save pipeline to output_dir.
    Returns metrics dict (accuracy, etc.).
    """
    settings = get_settings()
    output_dir = output_dir or settings.models_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    texts, y = load_training_data(data_path)
    if not texts:
        raise ValueError("No training data. Provide a CSV with 'text' and 'intent' or use defaults.")

    X_train, X_test, y_train, y_test = train_test_split(
        texts, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    score = pipeline.score(X_test, y_test)

    out_path = output_dir / "intent_model.joblib"
    joblib.dump(pipeline, out_path)

    return {
        "accuracy": float(score),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "model_path": str(out_path),
    }
