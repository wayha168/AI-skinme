"""FastAPI application factory."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from skin_assistant.api.routes import router

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
