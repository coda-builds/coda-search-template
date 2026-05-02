"""
Async database engine and session factory.

Uses SQLAlchemy 2.x async mode backed by asyncpg.
Every FastAPI route should depend on `get_db()` to get a scoped session.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy import text

from app.config import get_settings

logger = logging.getLogger(__name__)

_engine = None
_session_factory = None


def _get_engine():
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(
            settings.async_database_url,
            echo=not settings.is_production,   # SQL logging in dev only
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,                # Detect stale connections
        )
    return _engine


def _get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind=_get_engine(),
            expire_on_commit=False,
            autocommit=False,
            autoflush=False,
        )
    return _session_factory


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency — yields a database session and closes it on exit.

    Usage in a route::

        @router.get("/example")
        async def example(db: AsyncSession = Depends(get_db)):
            ...
    """
    async with _get_session_factory()() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def check_db_connection() -> bool:
    """Ping the database; returns True if reachable."""
    try:
        async with _get_session_factory()() as session:
            await session.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        logger.warning("Database health check failed: %s", exc)
        return False


@asynccontextmanager
async def lifespan_db():
    """
    Context manager for use in FastAPI lifespan — disposes the engine on shutdown.
    """
    try:
        yield
    finally:
        if _engine is not None:
            await _engine.dispose()
            logger.info("Database engine disposed.")
