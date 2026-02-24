"""Intent labels for chat classification (used by trainer and predictor)."""
INTENT_LABELS = [
    "greeting",
    "ingredient_info",      # what is X, tell me about X
    "products_with_ingredient",  # products containing X
    "product_recommendation",    # recommend for dry skin, I have acne
    "general_ingredient_search", # search ingredients
    "other",
]

INTENT_LABEL_TO_ID = {l: i for i, l in enumerate(INTENT_LABELS)}
ID_TO_INTENT_LABEL = {i: l for i, l in enumerate(INTENT_LABELS)}
