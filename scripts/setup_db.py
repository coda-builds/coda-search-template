#!/usr/bin/env python3
"""
Database setup script.

Run this once after creating your Supabase project:

    python scripts/setup_db.py

What it does:
  1. Enables the pgvector extension
  2. Creates the `items` table
  3. Creates an HNSW index on the embedding column for fast ANN search

HNSW index parameters (tune for your dataset):
  m              — number of bi-directional links per node (default 16)
                   Higher → better recall, more memory
  ef_construction— size of the dynamic search list used during index build (default 64)
                   Higher → better recall, slower build

Reference: https://github.com/pgvector/pgvector#hnsw
"""

import asyncio
import logging
import os
import sys

# Allow running from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from app.config import get_settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


async def setup() -> None:
    settings = get_settings()
    engine = create_async_engine(settings.async_database_url, echo=True)
    dims = settings.embedding_dimensions

    async with engine.begin() as conn:
        # 1. Enable pgvector
        logger.info("Enabling pgvector extension …")
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))

        # 2. Create items table
        logger.info("Creating items table …")
        await conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS items (
                id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                name        VARCHAR(255) NOT NULL,
                description TEXT         NOT NULL,
                metadata    JSONB        NOT NULL DEFAULT '{{}}',
                embedding   VECTOR({dims}),
                created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
                updated_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
            );
        """))

        # 3. Create HNSW index for cosine distance
        #
        # vector_cosine_ops is the correct operator class for cosine distance (<=>).
        # This index is used automatically when the query ORDER BY uses <=>.
        logger.info("Creating HNSW index (cosine distance) …")
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS items_embedding_hnsw_idx
            ON items
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64);
        """))

        # 4. GIN index on metadata for fast JSONB filtering
        logger.info("Creating GIN index on metadata …")
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS items_metadata_gin_idx
            ON items USING gin (metadata);
        """))

        # 5. Trigger to keep updated_at in sync
        await conn.execute(text("""
            CREATE OR REPLACE FUNCTION update_updated_at_column()
            RETURNS TRIGGER AS $$
            BEGIN
                NEW.updated_at = NOW();
                RETURN NEW;
            END;
            $$ language 'plpgsql';
        """))
        await conn.execute(text("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_trigger
                    WHERE tgname = 'set_updated_at' AND tgrelid = 'items'::regclass
                ) THEN
                    CREATE TRIGGER set_updated_at
                    BEFORE UPDATE ON items
                    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
                END IF;
            END;
            $$;
        """))

    await engine.dispose()
    logger.info("✅  Database setup complete.")


if __name__ == "__main__":
    asyncio.run(setup())
