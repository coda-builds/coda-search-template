"""
Embedding service — wraps sentence-transformers and exposes a clean async API.

The model is loaded once at application startup and reused for every request.
Encoding is synchronous (CPU-bound) so we run it in a thread-pool executor to
avoid blocking the async event loop.
"""

from __future__ import annotations

import asyncio
import logging
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import get_settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Thread-safe wrapper around SentenceTransformer.

    Usage::

        service = get_embedding_service()
        vector = await service.embed("some text")
        vectors = await service.embed_batch(["text1", "text2"])
    """

    def __init__(self) -> None:
        settings = get_settings()
        logger.info("Loading embedding model: %s", settings.embedding_model)
        self._model = SentenceTransformer(settings.embedding_model)
        self._batch_size = settings.embedding_batch_size
        # Warm-up: encode a dummy string so the first real request isn't slow.
        self._model.encode(["warmup"], batch_size=1, show_progress_bar=False)
        logger.info(
            "Embedding model ready — output dimensions: %d",
            self._model.get_sentence_embedding_dimension(),
        )

    def _encode_sync(self, texts: list[str]) -> list[list[float]]:
        """Synchronous encode — runs in a thread-pool executor."""
        vectors: np.ndarray = self._model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,   # L2-normalise → cosine = dot product
            convert_to_numpy=True,
        )
        return vectors.tolist()

    async def embed(self, text: str) -> list[float]:
        """Embed a single string asynchronously."""
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(None, self._encode_sync, [text])
        return results[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of strings asynchronously."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._encode_sync, texts)


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    """
    Cached singleton.  Call this everywhere — model is only loaded once.
    The lru_cache is process-scoped so it survives across requests.
    """
    return EmbeddingService()
