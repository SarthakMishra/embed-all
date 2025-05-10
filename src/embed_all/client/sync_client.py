"""
Synchronous client for Embed-All.

This module contains the synchronous client implementation.
"""

import logging
import warnings
from typing import cast

import httpx

from embed_all.client.base_client import BaseClient, EmbeddingKwargs
from embed_all.client.providers import get_provider_client
from embed_all.models import (
	BatchEmbeddingRequest,
	BatchEmbeddingResponse,
	ProviderModel,
	TextEmbeddingRequest,
	TextEmbeddingResponse,
)
from embed_all.models.errors import EmbedError

logger = logging.getLogger("embed_all")


class Client(BaseClient):
	"""Synchronous client for embedding operations.

	This client provides a synchronous interface for generating embeddings
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
		"""Initialize the synchronous client.

		Args:
			provider: The provider name
			api_key: API key for the provider
			base_url: Base URL for API requests
			model: Default model to use for embeddings
			timeout: Request timeout in seconds
			max_retries: Maximum number of retries for failed requests
		"""
		# Initialize provider_specific_config and _provider_client first
		provider_specific_config = ProviderModel(
			provider=provider,
			api_key=api_key,
			base_url=base_url,
			model=model,
			timeout=timeout,
			max_retries=max_retries,
		)
		self._provider_client = get_provider_client(provider)(provider_specific_config)

		super().__init__(
			provider=provider,
			api_key=api_key,
			base_url=base_url,
			model=model,
			timeout=timeout,
			max_retries=max_retries,
		)

		self._http_client = httpx.Client(
			timeout=timeout,
			event_hooks={"response": [self._log_response]},
			follow_redirects=True,
		)

	def _get_default_model(self) -> str:
		"""Return the default model for this provider."""
		return self._provider_client.default_model

	def _log_response(self, response: httpx.Response) -> None:
		"""Log the response for debugging purposes."""
		response.read()
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

		Args:
			text: The text or list of texts to embed
			model: The model to use (overrides the default)
			dimensions: Desired dimensionality of the embeddings
			**kwargs: Additional provider-specific parameters

		Returns:
			TextEmbeddingResponse containing the embeddings
		"""
		model_to_use: str = model or self.resolved_model_name
		request = TextEmbeddingRequest(
			model=model_to_use,
			input=text,
			dimensions=dimensions,
			**cast("dict", kwargs),
		)

		try:
			return self._provider_client.embed(request, self._http_client)
		except Exception as e:
			logger.exception("Error generating embeddings")
			if isinstance(e, EmbedError):
				raise
			msg = f"Error generating embeddings: {e!s}"
			raise EmbedError(
				msg,
				provider=self.config.provider,
			) from e

	def embed_batch(
		self,
		texts: list[str],
		model: str | None = None,
		dimensions: int | None = None,
		batch_size: int = 32,
		**kwargs: EmbeddingKwargs,
	) -> BatchEmbeddingResponse:
		"""Generate embeddings for a batch of texts.

		Args:
			texts: List of texts to embed
			model: The model to use (overrides the default)
			dimensions: Desired dimensionality of the embeddings
			batch_size: Number of texts to process in each batch
			**kwargs: Additional provider-specific parameters

		Returns:
			BatchEmbeddingResponse containing the embeddings
		"""
		model_to_use: str = model or self.resolved_model_name
		request = BatchEmbeddingRequest(
			model=model_to_use,
			inputs=texts,
			dimensions=dimensions,
			batch_size=batch_size,
			**cast("dict", kwargs),
		)

		try:
			return self._provider_client.embed_batch(request, self._http_client)
		except Exception as e:
			logger.exception("Error generating batch embeddings")
			if isinstance(e, EmbedError):
				raise
			msg = f"Error generating batch embeddings: {e!s}"
			raise EmbedError(
				msg,
				provider=self.config.provider,
			) from e

	async def aembed(
		self,
		text: str | list[str],
		model: str | None = None,
		dimensions: int | None = None,
		**kwargs: EmbeddingKwargs,
	) -> TextEmbeddingResponse:
		"""Asynchronously generate embeddings for text input (simulated).

		This method is an async-compatible wrapper around the synchronous `embed`
		method. It issues a UserWarning because it does not provide true
		non-blocking async behavior. For that, use `AsyncClient`.

		Args:
			text: The text or list of texts to embed
			model: The model to use (overrides the default)
			dimensions: Desired dimensionality of the embeddings
			**kwargs: Additional provider-specific parameters

		Returns:
			TextEmbeddingResponse containing the embeddings
		"""
		warnings.warn(
			"Running asynchronous `aembed` from synchronous Client. "
			"For true non-blocking async behavior, use `AsyncClient`.",
			UserWarning,
			stacklevel=2,
		)
		return self.embed(text, model, dimensions, **kwargs)

	async def aembed_batch(
		self,
		texts: list[str],
		model: str | None = None,
		dimensions: int | None = None,
		batch_size: int = 32,
		**kwargs: EmbeddingKwargs,
	) -> BatchEmbeddingResponse:
		"""Asynchronously generate embeddings for a batch of texts (simulated).

		This method is an async-compatible wrapper around the synchronous `embed_batch`
		method. It issues a UserWarning because it does not provide true
		non-blocking async behavior. For that, use `AsyncClient`.

		Args:
			texts: List of texts to embed
			model: The model to use (overrides the default)
			dimensions: Desired dimensionality of the embeddings
			batch_size: Number of texts to process in each batch
			**kwargs: Additional provider-specific parameters

		Returns:
			BatchEmbeddingResponse containing the embeddings
		"""
		warnings.warn(
			"Running asynchronous `aembed_batch` from synchronous Client. "
			"For true non-blocking async behavior, use `AsyncClient`.",
			UserWarning,
			stacklevel=2,
		)
		return self.embed_batch(texts, model, dimensions, batch_size, **kwargs)

	def close(self) -> None:
		"""Close the HTTP client and release resources."""
		self._http_client.close()

	def __enter__(self) -> "Client":
		"""Enter context manager."""
		return self

	def __exit__(self, *args: object) -> None:
		"""Exit context manager and close the client."""
		self.close()
