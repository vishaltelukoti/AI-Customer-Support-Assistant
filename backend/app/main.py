from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import health, tickets
from app.core.config import Settings


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings if settings is not None else Settings()
    application = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description="Day 1 foundation. Tickets are acknowledged, not classified or stored.",
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    application.include_router(health.router)
    application.include_router(tickets.router)
    return application


app = create_app()
