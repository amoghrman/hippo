"""OpenAI embedding implementation."""

from openai import AsyncOpenAI

from .base import Embedder


class OpenAIEmbedder(Embedder):
    """Embedder backed by OpenAI's text-embedding-3-small (1536 dims).

    Example::

        embedder = OpenAIEmbedder(api_key="sk-...")
        vec = await embedder.embed("User prefers dark mode")
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "text-embedding-3-small",
    ) -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model
        self._dims = 1536

    async def embed(self, text: str) -> list[float]:
        """Embed a single string.

        Example::

            vec = await embedder.embed("hello world")
        """
        response = await self._client.embeddings.create(input=text, model=self._model)
        return response.data[0].embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple strings in a single API call.

        Example::

            vecs = await embedder.embed_batch(["hello", "world"])
        """
        response = await self._client.embeddings.create(input=texts, model=self._model)
        ordered = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in ordered]

    @property
    def dimensions(self) -> int:
        return self._dims
