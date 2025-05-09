"""
Asynchronous client for Embed-All.

This module contains the asynchronous client implementation.
"""

import asyncio
import logging
from typing import cast

import httpx

from embed_all.client.base_client import BaseClient, EmbeddingKwargs
from embed_all.client.providers import get_provider_client
from embed_all.models import (
	BatchEmbeddingRequest,
	BatchEmbeddingResponse,
	TextEmbeddingRequest,
	TextEmbeddingResponse,
)
from embed_all.models.errors import EmbedError

logger = logging.getLogger("embed_all")


class AsyncClient(BaseClient):
	"""Asynchronous client for embedding operations.

	This client provides an asynchronous interface for generating embeddings
	from various providers.
	"""

	def __init__(
		self,
		provider: str,
		api_key: str | None = None,
		base_url: str | None = None,
		model: str | None = None,
		timeout: int = 60,
		max_retries: int = 3,
	) -> None:
		"""Initialize the asynchronous client.

		Args:
			provider: The provider name
			api_key: API key for the provider
			base_url: Base URL for API requests
			model: Default model to use for embeddings
			timeout: Request timeout in seconds
			max_retries: Maximum number of retries for failed requests
		"""
		super().__init__(
			provider=provider,
			api_key=api_key,
			base_url=base_url,
			model=model,
			timeout=timeout,
			max_retries=max_retries,
		)

		self._http_client = httpx.AsyncClient(
			timeout=timeout,
			event_hooks={"response": [self._log_response]},
			follow_redirects=True,
		)

		# Initialize the provider-specific client
		self._provider_client = get_provider_client(provider)(self.config)

	def _get_default_model(self) -> str:
		"""Return the default model for this provider."""
		return self._provider_client.default_model

	async def _log_response(self, response: httpx.Response) -> None:
		"""Log the response for debugging purposes."""
		logger.debug(
			f"Response from {response.request.url}: "
			f"status_code={response.status_code}, "
			f"elapsed={response.elapsed.total_seconds():.2f}s"
		)

	def embed(
		self,
		text: str | list[str],
		model: str | None = None,
		dimensions: int | None = None,
		**kwargs: EmbeddingKwargs,
	) -> TextEmbeddingResponse:
		"""Generate embeddings for text input.

		This method is implemented for API compatibility but uses the asynchronous
		implementation under the hood by running it in the event loop.

		Args:
			text: The text or list of texts to embed
			model: The model to use (overrides the default)
			dimensions: Desired dimensionality of the embeddings
			**kwargs: Additional provider-specific parameters

		Returns:
			TextEmbeddingResponse containing the embeddings
		"""
		logger.warning("Using async client with sync method. For better control, use aembed for async operations.")

		loop = asyncio.get_event_loop()
		return loop.run_until_complete(self.aembed(text, model, dimensions, **kwargs))

	def embed_batch(
		self,
		texts: list[str],
		model: str | None = None,
		dimensions: int | None = None,
		batch_size: int = 32,
		**kwargs: EmbeddingKwargs,
	) -> BatchEmbeddingResponse:
		"""Generate embeddings for a batch of texts.

		This method is implemented for API compatibility but uses the asynchronous
		implementation under the hood by running it in the event loop.

		Args:
			texts: List of texts to embed
			model: The model to use (overrides the default)
			dimensions: Desired dimensionality of the embeddings
			batch_size: Number of texts to process in each batch
			**kwargs: Additional provider-specific parameters

		Returns:
			BatchEmbeddingResponse containing the embeddings
		"""
		logger.warning(
			"Using async client with sync method. For better control, use aembed_batch for async operations."
		)

		loop = asyncio.get_event_loop()
		return loop.run_until_complete(self.aembed_batch(texts, model, dimensions, batch_size, **kwargs))

	async def aembed(
		self,
		text: str | list[str],
		model: str | None = None,
		dimensions: int | None = None,
		**kwargs: EmbeddingKwargs,
	) -> TextEmbeddingResponse:
		"""Asynchronously generate embeddings for text input.

		Args:
			text: The text or list of texts to embed
			model: The model to use (overrides the default)
			dimensions: Desired dimensionality of the embeddings
			**kwargs: Additional provider-specific parameters

		Returns:
			TextEmbeddingResponse containing the embeddings
		"""
		request = TextEmbeddingRequest(
			model=model or self.config.model,
			input=text,
			dimensions=dimensions,
			**cast("dict", kwargs),
		)

		try:
			return await self._provider_client.aembed(request, self._http_client)
		except Exception as e:
			logger.exception("Error generating embeddings")
			if isinstance(e, EmbedError):
				raise
			msg = f"Error generating embeddings: {e!s}"
			raise EmbedError(
				msg,
				provider=self.config.provider,
			) from e

	async def aembed_batch(
		self,
		texts: list[str],
		model: str | None = None,
		dimensions: int | None = None,
		batch_size: int = 32,
		**kwargs: EmbeddingKwargs,
	) -> BatchEmbeddingResponse:
		"""Asynchronously generate embeddings for a batch of texts.

		Args:
			texts: List of texts to embed
			model: The model to use (overrides the default)
			dimensions: Desired dimensionality of the embeddings
			batch_size: Number of texts to process in each batch
			**kwargs: Additional provider-specific parameters

		Returns:
			BatchEmbeddingResponse containing the embeddings
		"""
		request = BatchEmbeddingRequest(
			model=model or self.config.model,
			inputs=texts,
			dimensions=dimensions,
			batch_size=batch_size,
			**cast("dict", kwargs),
		)

		try:
			return await self._provider_client.aembed_batch(request, self._http_client)
		except Exception as e:
			logger.exception("Error generating batch embeddings")
			if isinstance(e, EmbedError):
				raise
			msg = f"Error generating batch embeddings: {e!s}"
			raise EmbedError(
				msg,
				provider=self.config.provider,
			) from e

	async def close(self) -> None:
		"""Close the HTTP client and release resources."""
		await self._http_client.aclose()

	async def __aenter__(self) -> "AsyncClient":
		"""Enter async context manager."""
		return self

	async def __aexit__(self, *args: object) -> None:
		"""Exit async context manager and close the client."""
		await self.close()
