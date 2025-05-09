"""
Embed-All: A light type-safe wrapper around embedding endpoints.

This library provides a unified interface to various embedding model APIs
from different providers, with minimal dependencies.
"""

import logging

from embed_all.client import AsyncClient, Client
from embed_all.client.base_client import EmbeddingKwargs

__version__ = "0.1.0"

# Set up logging
logging.getLogger("embed_all").addHandler(logging.NullHandler())


def embed(
	text: str | list[str],
	provider: str = "voyage",
	api_key: str | None = None,
	model: str | None = None,
	dimensions: int | None = None,
	**kwargs: EmbeddingKwargs,
) -> list[list[float]]:
	"""Generate embeddings for text input using the specified provider.

	A convenience function that creates a client and generates embeddings.

	Args:
		text: The text or list of texts to embed
		provider: The provider to use
		api_key: API key for the provider
		model: The model to use
		dimensions: Desired dimensionality of the embeddings
		**kwargs: Additional provider-specific parameters

	Returns:
		List of embedding vectors as lists of floats
	"""
	with Client(provider=provider, api_key=api_key, model=model) as client:
		response = client.embed(text, model=model, dimensions=dimensions, **kwargs)
		return response.to_list()


__all__ = ["AsyncClient", "Client", "__version__", "embed"]
