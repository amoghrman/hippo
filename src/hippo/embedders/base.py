"""Abstract base class for text embedders."""
from abc import ABC, abstractmethod


class Embedder(ABC):
    """Pluggable text-to-vector embedder.

    Implement this to swap out OpenAI for any other embedding provider.

    Example::

        class MyEmbedder(Embedder):
            async def embed(self, text: str) -> list[float]:
                return my_model.encode(text).tolist()

            async def embed_batch(self, texts: list[str]) -> list[list[float]]:
                return [my_model.encode(t).tolist() for t in texts]

            @property
            def dimensions(self) -> int:
                return 768
    """

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Embed a single string into a float vector.

        Example::

            vec = await embedder.embed("User prefers dark mode")
        """
        ...

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple strings in one call (more efficient than looping embed).

        Example::

            vecs = await embedder.embed_batch(["hello", "world"])
        """
        ...

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Number of dimensions in the output vector."""
        ...
