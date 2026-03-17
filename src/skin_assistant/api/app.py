"""FastAPI application factory."""
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from skin_assistant.api.routes import router

# Load .env when API is started directly (e.g. uvicorn skin_assistant.api.app:app)
try:
    from dotenv import load_dotenv
    _root = Path(__file__).resolve().parents[3]  # api -> skin_assistant -> src -> project root
    load_dotenv(_root / ".env")
except Exception:
    pass

def create_app() -> FastAPI:
    app = FastAPI(
        title="Skin Assistant API",
        description="API for skincare ingredient lookup and product recommendations. Integrate with your backend via REST.",
        version="0.2.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app


app = create_app()
