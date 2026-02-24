"""
ML models for Skin Assistant: intent classification and optional semantic search.
"""
from skin_assistant.models.intent_labels import INTENT_LABELS
from skin_assistant.models.predictor import IntentPredictor

__all__ = ["INTENT_LABELS", "IntentPredictor"]
