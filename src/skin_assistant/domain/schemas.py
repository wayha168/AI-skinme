"""Domain schemas (DTOs) for API request/response."""
from pydantic import BaseModel, Field
from typing import Optional


class Message(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    history: list[Message] = Field(default_factory=list, max_length=20)
    use_llm: bool = True


class ChatResponse(BaseModel):
    reply: str


# --- Backend integration (e.g. Spring): save to database ---

class ChatLogRequest(BaseModel):
    """Payload for logging a chat turn so backend can persist to DB."""
    session_id: str = Field(..., min_length=1, max_length=128)
    user_id: Optional[str] = Field(None, max_length=128)
    message: str = Field(..., min_length=1, max_length=2000)
    reply: str = Field(..., max_length=10000)
    timestamp: Optional[str] = None  # ISO datetime; backend can set if omitted


class FeedbackRequest(BaseModel):
    """Payload for saving user feedback so backend can persist to DB."""
    session_id: str = Field(..., min_length=1, max_length=128)
    message_id: Optional[str] = Field(None, max_length=128)
    rating: Optional[int] = Field(None, ge=1, le=5)  # 1-5 or use thumbs up/down
    thumbs_up: Optional[bool] = None
    comment: Optional[str] = Field(None, max_length=1000)


class SaveResponse(BaseModel):
    """Response for save endpoints (forwarded to Spring)."""
    saved: bool = True
    message: str = "ok"
    backend_saved: Optional[bool] = None  # True if Spring acknowledged


class IngredientOut(BaseModel):
    name: Optional[str] = None
    scientific_name: Optional[str] = None
    what_is_it: Optional[str] = None
    what_does_it_do: Optional[str] = None
    who_is_it_good_for: Optional[str] = None
    who_should_avoid: Optional[str] = None
    url: Optional[str] = None


class ProductOut(BaseModel):
    product_name: Optional[str] = None
    product_type: Optional[str] = None
    product_url: Optional[str] = None
    price: Optional[str] = None


class SearchIngredientsResponse(BaseModel):
    query: str
    count: int
    ingredients: list[IngredientOut]


class SearchProductsResponse(BaseModel):
    query: str
    count: int
    products: list[ProductOut]
