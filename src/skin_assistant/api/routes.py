"""FastAPI route handlers for Skin Assistant API."""
from typing import Optional

import requests
from fastapi import APIRouter, HTTPException, Query

from skin_assistant.config import get_settings
from skin_assistant.domain.schemas import (
    ChatRequest,
    ChatResponse,
    ChatLogRequest,
    FeedbackRequest,
    SaveResponse,
    IngredientOut,
    ProductOut,
    SearchIngredientsResponse,
    SearchProductsResponse,
)
from skin_assistant.infrastructure import KnowledgeRepository
from skin_assistant.services import ChatService

router = APIRouter(prefix="/v1", tags=["skin-assistant"])
_repo = KnowledgeRepository()
_chat = ChatService(repo=_repo)


def _forward_to_backend(path: str, payload: dict) -> bool:
    """POST payload to Spring (or other) backend for saving to database. Returns True if backend responded 2xx."""
    base = get_settings().backend_url
    if not base:
        return False
    url = f"{base}{path}" if path.startswith("/") else f"{base}/{path}"
    try:
        r = requests.post(url, json=payload, timeout=5)
        return 200 <= r.status_code < 300
    except Exception:
        return False


def _to_ingredient_out(d: dict) -> IngredientOut:
    return IngredientOut(
        name=d.get("name"),
        scientific_name=d.get("scientific_name"),
        what_is_it=d.get("what_is_it"),
        what_does_it_do=d.get("what_does_it_do"),
        who_is_it_good_for=d.get("who_is_it_good_for"),
        who_should_avoid=d.get("who_should_avoid"),
        url=d.get("url"),
    )


def _to_product_out(d: dict) -> ProductOut:
    return ProductOut(
        product_name=d.get("product_name"),
        product_type=d.get("product_type"),
        product_url=d.get("product_url"),
        price=d.get("price"),
    )


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """Send a message and get the assistant reply. Optional history for context."""
    history = [{"role": m.role, "content": m.content} for m in req.history]
    reply = _chat.get_reply(req.message, conversation_history=history, use_llm=req.use_llm)
    return ChatResponse(reply=reply)


@router.get("/ingredients/search", response_model=SearchIngredientsResponse)
def search_ingredients(
    q: str = Query(..., min_length=1, max_length=200),
    top_k: int = Query(5, ge=1, le=20),
) -> SearchIngredientsResponse:
    """Search ingredients by name or description."""
    results = _repo.search_ingredients(q, top_k=top_k)
    return SearchIngredientsResponse(
        query=q,
        count=len(results),
        ingredients=[_to_ingredient_out(r) for r in results],
    )


@router.get("/ingredients/{name}", response_model=IngredientOut)
def get_ingredient(name: str) -> IngredientOut:
    """Get a single ingredient by name."""
    ing = _repo.get_ingredient_by_name(name)
    if not ing:
        raise HTTPException(status_code=404, detail="Ingredient not found")
    return _to_ingredient_out(ing)


@router.get("/products", response_model=SearchProductsResponse)
def search_products(
    concern: Optional[str] = Query(None, max_length=200),
    product_type: Optional[str] = Query(None, max_length=100),
    ingredient: Optional[str] = Query(None, max_length=200),
    top_k: int = Query(5, ge=1, le=20),
) -> SearchProductsResponse:
    """
    Search products by concern (e.g. dry skin) or by ingredient name.
    Use `concern` for recommendations, or `ingredient` for products containing an ingredient.
    """
    if ingredient:
        results = _repo.get_products_containing_ingredient(ingredient, top_k=top_k)
        q = ingredient
    elif concern:
        results = _repo.search_products_by_concern(concern, product_type=product_type, top_k=top_k)
        q = concern
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide either 'concern' or 'ingredient' query parameter.",
        )
    return SearchProductsResponse(query=q, count=len(results), products=[_to_product_out(r) for r in results])


@router.get("/intent")
def predict_intent(q: str = Query(..., min_length=1, max_length=500)) -> dict:
    """Predict intent of a message (e.g. greeting, ingredient_info). Requires a trained model in models/artifacts."""
    try:
        from skin_assistant.models import IntentPredictor
        predictor = IntentPredictor()
        intent = predictor.predict(q)
        proba = predictor.predict_proba(q)
        return {"query": q, "intent": intent, "probabilities": proba}
    except Exception:
        return {"query": q, "intent": "other", "probabilities": {}}


# --- Backend integration (Spring): save to database ---

@router.post("/chat/log", response_model=SaveResponse)
def chat_log(req: ChatLogRequest) -> SaveResponse:
    """
    Log a chat turn for persistence in your backend database.
    Spring can expose POST /api/v1/chat/log to receive this payload and save to DB.
    If SPRING_BACKEND_URL is set, the payload is also forwarded to {url}/api/v1/chat/log.
    """
    payload = req.model_dump()
    backend_saved = _forward_to_backend("/api/v1/chat/log", payload)
    return SaveResponse(saved=True, backend_saved=backend_saved if get_settings().backend_url else None)


@router.post("/feedback", response_model=SaveResponse)
def feedback(req: FeedbackRequest) -> SaveResponse:
    """
    Save user feedback for persistence in your backend database.
    Spring can expose POST /api/v1/feedback to receive this payload and save to DB.
    If SPRING_BACKEND_URL is set, the payload is also forwarded to {url}/api/v1/feedback.
    """
    payload = req.model_dump()
    backend_saved = _forward_to_backend("/api/v1/feedback", payload)
    return SaveResponse(saved=True, backend_saved=backend_saved if get_settings().backend_url else None)


@router.get("/routes")
def list_routes() -> dict:
    """
    List route paths for backend (e.g. Spring) integration.
    Use these paths when calling this API or when implementing endpoints that receive forwarded payloads.
    """
    base = "/v1"
    return {
        "base": base,
        "chat": {
            "post_chat": f"POST {base}/chat",
            "post_chat_log": f"POST {base}/chat/log",
        },
        "feedback": {"post_feedback": f"POST {base}/feedback"},
        "ingredients": {
            "search": f"GET {base}/ingredients/search?q=...",
            "get_by_name": f"GET {base}/ingredients/{{name}}",
        },
        "products": {"search": f"GET {base}/products?concern=...|ingredient=..."},
        "intent": {"predict": f"GET {base}/intent?q=..."},
        "health": f"GET {base}/health",
    }


@router.get("/health")
def health() -> dict:
    """Health check for load balancers and monitoring."""
    return {"status": "ok", "service": "skin-assistant"}
