"""Local sentence-transformers embedder — no API key required."""

from __future__ import annotations

import asyncio

from .base import Embedder

_KNOWN_DIMS: dict[str, int] = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "all-MiniLM-L6-v2": 384,
    "all-MiniLM-L12-v2": 384,
    "all-mpnet-base-v2": 768,
}


class SentenceTransformersEmbedder(Embedder):
    """Local embedder using sentence-transformers (no API key needed).

    The model is loaded lazily on first use. For well-known models the
    ``dimensions`` property is available immediately via a lookup table, so
    ``Hippo.setup()`` can create the correct vector column without triggering
    a model download.

    Example::

        embedder = SentenceTransformersEmbedder("BAAI/bge-small-en-v1.5")
        await hippo.setup()  # uses 384-dim column — no model download yet
        vec = await embedder.embed("hello world")  # model loads here
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self._model_name = model_name
        self._model = None
        self._dims: int | None = _KNOWN_DIMS.get(model_name)

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
            self._dims = self._model.get_embedding_dimension()
        return self._model

    @property
    def dimensions(self) -> int:
        if self._dims is not None:
            return self._dims
        return self._load().get_embedding_dimension()

    async def embed(self, text: str) -> list[float]:
        loop = asyncio.get_running_loop()
        model = self._load()
        vec = await loop.run_in_executor(
            None, lambda: model.encode(text, normalize_embeddings=True)
        )
        return vec.tolist()

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        loop = asyncio.get_running_loop()
        model = self._load()
        vecs = await loop.run_in_executor(
            None, lambda: model.encode(texts, normalize_embeddings=True)
        )
        return [v.tolist() for v in vecs]
