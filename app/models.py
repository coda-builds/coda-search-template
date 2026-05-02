"""
Database model (SQLAlchemy) and API schemas (Pydantic).

The `Item` table is intentionally generic — replace `name`, `description`,
and `metadata` with whatever fields your dataset requires.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pgvector.sqlalchemy import Vector
from pydantic import BaseModel, Field
from sqlalchemy import DateTime, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.config import get_settings


# ── SQLAlchemy ORM ────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


class Item(Base):
    """
    Generic searchable item.  Works for products, articles, documents, etc.
    The `embedding` column stores a 384-dim vector (all-MiniLM-L6-v2).
    The HNSW index on this column is created by the setup script.
    """

    __tablename__ = "items"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    # Free-form metadata: category, price, tags — anything your domain needs.
    metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    embedding: Mapped[list[float]] = mapped_column(
        Vector(get_settings().embedding_dimensions), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=lambda: datetime.now(timezone.utc),
    )


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class ItemCreate(BaseModel):
    """Payload accepted by POST /ingest/item."""

    name: str = Field(..., min_length=1, max_length=255, examples=["Wireless Keyboard"])
    description: str = Field(
        ..., min_length=1, examples=["Compact tenkeyless mechanical keyboard with RGB backlight."]
    )
    metadata: dict[str, Any] = Field(default_factory=dict, examples=[{"category": "Electronics", "price": 79.99}])


class ItemBatchCreate(BaseModel):
    """Payload accepted by POST /ingest/batch."""

    items: list[ItemCreate] = Field(..., min_length=1, max_length=500)


class ItemOut(BaseModel):
    """Item representation returned by the API."""

    model_config = {"from_attributes": True}

    id: uuid.UUID
    name: str
    description: str
    metadata: dict[str, Any]
    created_at: datetime


class SearchRequest(BaseModel):
    """Payload accepted by POST /search."""

    query: str = Field(..., min_length=1, max_length=1000, examples=["lightweight running shoes"])
    top_k: int = Field(default=10, ge=1, le=100)
    similarity_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    # Optional metadata filter — e.g. {"category": "Footwear"}
    filters: dict[str, Any] | None = Field(default=None)


class SearchResult(BaseModel):
    """Single result row returned by the search endpoint."""

    item: ItemOut
    similarity: float = Field(..., description="Cosine similarity score (0–1, higher is better)")


class SearchResponse(BaseModel):
    """Complete search response envelope."""

    query: str
    results: list[SearchResult]
    total: int
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    environment: str
    model: str
    db_connected: bool
