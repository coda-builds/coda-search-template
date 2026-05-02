"""
Search router — POST /search

Accepts a natural-language query, converts it to an embedding, and returns
the most similar items from the database using pgvector HNSW cosine search.
"""

from __future__ import annotations

import logging
import re
import time

from fastapi import APIRouter, Depends, Header, HTTPException, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import get_db
from app.embeddings import get_embedding_service
from app.models import ItemOut, SearchRequest, SearchResponse, SearchResult

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/search", tags=["Search"])

# Compiled once at import time — used to validate JSONB filter keys before
# they are interpolated into SQL. Keys must be alphanumeric + underscore only.
_SAFE_FILTER_KEY = re.compile(r"^[a-zA-Z0-9_]{1,64}$")


def _require_api_key(x_api_key: str = Header(...)) -> None:
    """Simple API key guard.  Replace with OAuth / JWT for multi-tenant use."""
    if x_api_key != get_settings().api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
        )


@router.post(
    "",
    response_model=SearchResponse,
    summary="Semantic search",
    description=(
        "Convert the query to an embedding and retrieve the most similar items "
        "using pgvector HNSW approximate nearest-neighbour search."
    ),
)
async def search(
    payload: SearchRequest,
    db: AsyncSession = Depends(get_db),
    _: None = Depends(_require_api_key),
) -> SearchResponse:
    t0 = time.perf_counter()

    # 1. Embed the query
    embedding_service = get_embedding_service()
    query_vector = await embedding_service.embed(payload.query)

    # 2. Build the SQL query
    #
    # pgvector operators:
    #   <=>  cosine distance   (0 = identical, 2 = opposite)
    #   1 - (a <=> b)  →  cosine *similarity* (1 = identical, −1 = opposite)
    #
    # Because we L2-normalise embeddings at encode time, cosine similarity
    # equals the dot product, but we keep the explicit formula for clarity.
    #
    # The HNSW index on the `embedding` column is used automatically by the
    # planner when the ORDER BY clause uses <=> on that column.

    # Build an optional JSONB filter clause
    filter_clause = ""
    filter_params: dict = {}
    if payload.filters:
        # Validate keys: only allow alphanumeric characters and underscores.
        # Keys are embedded into the SQL string (they cannot be parameterised
        # in PostgreSQL's JSONB ->> operator), so we must reject anything that
        # could break the query.
        conditions = []
        for i, (key, value) in enumerate(payload.filters.items()):
            if not _SAFE_FILTER_KEY.match(key):
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Filter key {key!r} contains invalid characters. "
                           "Only letters, digits, and underscores are allowed.",
                )
            param_name = f"filter_val_{i}"
            conditions.append(f"metadata->>{key!r} = :{param_name}")
            filter_params[param_name] = str(value)
        if conditions:
            filter_clause = "AND " + " AND ".join(conditions)

    sql = text(
        f"""
        SELECT
            id,
            name,
            description,
            metadata,
            created_at,
            1 - (embedding <=> CAST(:query_vector AS vector)) AS similarity
        FROM items
        WHERE embedding IS NOT NULL
          AND 1 - (embedding <=> CAST(:query_vector AS vector)) >= :threshold
          {filter_clause}
        ORDER BY embedding <=> CAST(:query_vector AS vector)
        LIMIT :top_k
        """
    )

    result = await db.execute(
        sql,
        {
            "query_vector": str(query_vector),
            "threshold": payload.similarity_threshold,
            "top_k": payload.top_k,
            **filter_params,
        },
    )
    rows = result.fetchall()

    latency_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "Search query=%r  results=%d  latency=%.1fms",
        payload.query,
        len(rows),
        latency_ms,
    )

    search_results = [
        SearchResult(
            item=ItemOut(
                id=row.id,
                name=row.name,
                description=row.description,
                metadata=row.metadata or {},
                created_at=row.created_at,
            ),
            similarity=round(float(row.similarity), 4),
        )
        for row in rows
    ]

    return SearchResponse(
        query=payload.query,
        results=search_results,
        total=len(search_results),
        latency_ms=round(latency_ms, 2),
    )
