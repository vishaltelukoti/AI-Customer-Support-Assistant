from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import time

from app.api.routes import health, tickets
from app.core.config import Settings
from app.monitoring.metrics import monitor


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

    @application.middleware("http")
    async def monitoring_middleware(request, call_next):
        started = time.perf_counter()
        error = False
        try:
            response = await call_next(request)
            error = response.status_code >= 500
            return response
        except Exception:
            error = True
            raise
        finally:
            monitor.record_request(time.perf_counter() - started, error=error)

    application.include_router(health.router)
    application.include_router(tickets.router)
    return application


app = create_app()
