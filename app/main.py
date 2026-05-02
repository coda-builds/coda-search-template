"""
coda-search-template — FastAPI application entry point.

Startup sequence:
  1. Load config
  2. Warm up embedding model (runs in background task)
  3. Connect to database
  4. Register routes
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.database import check_db_connection, lifespan_db
from app.embeddings import get_embedding_service
from app.models import HealthResponse
from app.routes.ingest import router as ingest_router
from app.routes.search import router as search_router

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG if not get_settings().is_production else logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once on startup, then again (after yield) on shutdown.
    Heavy initialisation goes here so it doesn't slow down the first request.
    """
    logger.info("Starting coda-search-template (env=%s)", get_settings().environment)

    # Warm up the embedding model synchronously in the main process.
    # This ensures the model is cached before the first request arrives.
    get_embedding_service()

    async with lifespan_db():
        yield

    logger.info("Shutdown complete.")


# ── Application ───────────────────────────────────────────────────────────────
def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Coda Search",
        description=(
            "Production-ready semantic search service. "
            "Powered by all-MiniLM-L6-v2 embeddings and pgvector HNSW indexing."
        ),
        version="1.0.0",
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
        lifespan=lifespan,
    )

    # CORS — tighten origins for production
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if not settings.is_production else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routes ────────────────────────────────────────────────────────────────
    app.include_router(search_router)
    app.include_router(ingest_router)

    # ── Health check (no auth required) ──────────────────────────────────────
    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health() -> HealthResponse:
        db_ok = await check_db_connection()
        return HealthResponse(
            status="ok" if db_ok else "degraded",
            environment=settings.environment,
            model=settings.embedding_model,
            db_connected=db_ok,
        )

    @app.get("/", include_in_schema=False)
    async def root():
        return JSONResponse({"service": "coda-search", "docs": "/docs"})

    return app


app = create_app()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=get_settings().port,
        reload=not get_settings().is_production,
        workers=1,   # Keep 1 worker: the model is loaded in-process
    )
