import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import health, monitoring, tickets
from app.core.config import Settings
from app.monitoring.metrics import monitor
from app.services import ticket_service

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI):
    processor = ticket_service.initialize_ticket_processor()
    application.state.ticket_processor = processor
    yield


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings if settings is not None else Settings()
    application = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description=(
            "POC support assistant with local classification, similar-ticket retrieval, "
            "RAG/agent routing, input/output filtering, explanations and lightweight monitoring."
        ),
        lifespan=lifespan,
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
            error = response.status_code >= 400
            return response
        except Exception:
            logger.exception("HTTP request handling failed")
            error = True
            raise
        finally:
            monitor.record_request(time.perf_counter() - started, error=error)

    application.include_router(health.router)
    application.include_router(monitoring.router)
    application.include_router(tickets.router)
    return application


app = create_app()
