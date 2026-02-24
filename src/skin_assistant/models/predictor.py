"""Load trained intent model and predict."""
from pathlib import Path
from typing import Optional

import joblib

from skin_assistant.config import get_settings
from skin_assistant.models.intent_labels import ID_TO_INTENT_LABEL, INTENT_LABELS


class IntentPredictor:
    """Predict chat intent from user message using trained pipeline."""

    def __init__(self, model_path: Optional[Path] = None):
        settings = get_settings()
        self._path = model_path or (settings.models_dir / "intent_model.joblib")
        self._pipeline = None

    def _load(self):
        if self._pipeline is None and self._path.exists():
            self._pipeline = joblib.load(self._path)

    def predict(self, text: str) -> str:
        """Return intent label for the given message."""
        self._load()
        if self._pipeline is None:
            return "other"
        label_id = self._pipeline.predict([text.strip()])[0]
        return ID_TO_INTENT_LABEL.get(int(label_id), "other")

    def predict_proba(self, text: str) -> dict[str, float]:
        """Return intent probabilities (if model supports it)."""
        self._load()
        if self._pipeline is None:
            return {l: 0.0 for l in INTENT_LABELS}
        probs = self._pipeline.predict_proba([text.strip()])[0]
        return {ID_TO_INTENT_LABEL.get(i, "other"): float(p) for i, p in enumerate(probs)}
