"""
Ingest router — POST /ingest/item  |  POST /ingest/batch  |  DELETE /ingest/{id}

Accepts items, generates embeddings, and upserts them into the database.
This is a write-path endpoint; protect it behind your API key.
"""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Depends, Header, HTTPException, status
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import get_db
from app.embeddings import get_embedding_service
from app.models import Item, ItemBatchCreate, ItemCreate, ItemOut

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ingest", tags=["Ingest"])


def _require_api_key(x_api_key: str = Header(...)) -> None:
    if x_api_key != get_settings().api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
        )


@router.post(
    "/item",
    response_model=ItemOut,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest a single item",
)
async def ingest_item(
    payload: ItemCreate,
    db: AsyncSession = Depends(get_db),
    _: None = Depends(_require_api_key),
) -> ItemOut:
    """Embed and store a single item."""
    embedding_service = get_embedding_service()
    vector = await embedding_service.embed(payload.description)

    item = Item(
        name=payload.name,
        description=payload.description,
        metadata=payload.metadata,
        embedding=vector,
    )
    db.add(item)
    await db.flush()       # populate item.id before refresh
    await db.refresh(item)
    logger.info("Ingested item id=%s name=%r", item.id, item.name)
    return ItemOut.model_validate(item)


@router.post(
    "/batch",
    status_code=status.HTTP_201_CREATED,
    summary="Ingest a batch of items (max 500)",
    response_model=dict,
)
async def ingest_batch(
    payload: ItemBatchCreate,
    db: AsyncSession = Depends(get_db),
    _: None = Depends(_require_api_key),
) -> dict:
    """
    Embed and store up to 500 items in one request.
    Embeddings are generated in a single batched forward pass for efficiency.
    """
    embedding_service = get_embedding_service()
    texts = [item.description for item in payload.items]
    vectors = await embedding_service.embed_batch(texts)

    db_items = [
        Item(
            name=item.name,
            description=item.description,
            metadata=item.metadata,
            embedding=vector,
        )
        for item, vector in zip(payload.items, vectors)
    ]
    db.add_all(db_items)
    await db.flush()

    logger.info("Ingested batch of %d items", len(db_items))
    return {"ingested": len(db_items), "ids": [str(i.id) for i in db_items]}


@router.delete(
    "/{item_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete an item by ID",
)
async def delete_item(
    item_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    _: None = Depends(_require_api_key),
) -> None:
    result = await db.execute(
        delete(Item).where(Item.id == item_id).returning(Item.id)
    )
    if result.rowcount == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Item {item_id} not found.",
        )
    logger.info("Deleted item id=%s", item_id)
