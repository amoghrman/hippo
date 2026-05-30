"""Tests for pluggable embedders.

The SentenceTransformers tests are marked slow because they download/load a
model on first run. Skip them with: pytest -m "not slow"
"""

import pytest

from hippo.embedders.sentence_transformers import _KNOWN_DIMS, SentenceTransformersEmbedder


def test_known_dims_lookup() -> None:
    """Dimensions are available before the model loads for well-known models."""
    for model_name, expected_dim in _KNOWN_DIMS.items():
        embedder = SentenceTransformersEmbedder(model_name)
        assert embedder.dimensions == expected_dim, (
            f"{model_name}: expected {expected_dim}, got {embedder.dimensions}"
        )
        assert embedder._model is None, "Model must not load just from dimensions property"


@pytest.mark.slow
@pytest.mark.asyncio
async def test_sentence_transformers_embed_dimension() -> None:
    """embed() returns a vector of the correct dimension."""
    embedder = SentenceTransformersEmbedder("BAAI/bge-small-en-v1.5")
    vec = await embedder.embed("hello world")
    assert len(vec) == 384
    assert embedder.dimensions == 384


@pytest.mark.slow
@pytest.mark.asyncio
async def test_sentence_transformers_embed_batch() -> None:
    """embed_batch() returns one vector per input text."""
    embedder = SentenceTransformersEmbedder("BAAI/bge-small-en-v1.5")
    texts = ["hello", "world", "foo bar"]
    vecs = await embedder.embed_batch(texts)
    assert len(vecs) == len(texts)
    assert all(len(v) == 384 for v in vecs)


@pytest.mark.slow
@pytest.mark.asyncio
async def test_sentence_transformers_normalized() -> None:
    """Embeddings are L2-normalized (magnitude ≈ 1.0)."""
    import math

    embedder = SentenceTransformersEmbedder("BAAI/bge-small-en-v1.5")
    vec = await embedder.embed("normalization check")
    magnitude = math.sqrt(sum(x * x for x in vec))
    assert abs(magnitude - 1.0) < 1e-4, f"Expected unit vector, got magnitude {magnitude}"
