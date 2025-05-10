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
from embed_all.models.base import ProviderModel
from embed_all.models.errors import EmbedError

# Store real httpx.AsyncClient type to use in isinstance checks,
# as httpx.AsyncClient might be mocked in tests.
_RealHttpxAsyncClient = httpx.AsyncClient

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
		# Step 1: Initialize provider_client.
		# Its config will have model=None if the user passed None.
		# BaseClient's __init__ will later resolve self.config.model to a string.
		provider_specific_config = ProviderModel(
			provider=provider,
			api_key=api_key,
			base_url=base_url,
			model=model,  # Pass the original model (can be None)
			timeout=timeout,
			max_retries=max_retries,
		)
		self._provider_client = get_provider_client(provider)(provider_specific_config)

		# Step 2: Call super().__init__().
		# BaseClient.__init__ will set self.config.
		# self.config.model will be resolved to str (model or default_model).
		super().__init__(
			provider=provider,
			api_key=api_key,
			base_url=base_url,
			model=model,  # Pass original model (str | None)
			timeout=timeout,
			max_retries=max_retries,
		)

		self._http_client = httpx.AsyncClient(
			timeout=timeout,
			event_hooks={"response": [self._log_response]},
			follow_redirects=True,
		)

	def _get_default_model(self) -> str:
		"""Return the default model for this provider."""
		return self._provider_client.default_model

	async def _log_response(self, response: httpx.Response) -> None:
		"""Log the response for debugging purposes."""
		await response.aread()  # Ensure response is read before accessing .elapsed
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

		# loop = asyncio.new_event_loop()
		# asyncio.set_event_loop(loop)

		# This approach with run_until_complete on a potentially existing loop is problematic.
		# If called from an async context (e.g., another async def function, or pytest-asyncio test),
		# get_running_loop() will succeed, but run_until_complete will fail as the loop is already running.
		# A true sync wrapper typically uses asyncio.run() to manage its own event loop lifecycle.
		# return loop.run_until_complete(self.aembed(text, model, dimensions, **kwargs))
		return asyncio.run(self.aembed(text, model, dimensions, **kwargs))

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

		# loop = asyncio.new_event_loop()
		# asyncio.set_event_loop(loop)

		# Similar to embed(), this is problematic if called from an existing async context.
		# Using asyncio.run() is generally safer for a sync wrapper.
		# return loop.run_until_complete(
		#     self.aembed_batch(texts, model, dimensions, batch_size, **kwargs)
		# )
		return asyncio.run(self.aembed_batch(texts, model, dimensions, batch_size, **kwargs))

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
		# Use the new property for a guaranteed string model name
		model_to_use: str = model or self.resolved_model_name
		request = TextEmbeddingRequest(
			model=model_to_use,
			input=text,
			dimensions=dimensions,
			**cast("dict", kwargs),
		)

		try:
			return await self._provider_client.aembed(request, cast("httpx.AsyncClient", self._http_client))
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
		model_to_use: str = model or self.resolved_model_name
		all_embeddings = []
		failed_indices = []
		total_usage = {}
		batch_count = 0
		num_texts = len(texts)
		for i in range(0, num_texts, batch_size):
			batch_texts = texts[i : i + batch_size]
			batch_indices = list(range(i, i + len(batch_texts)))
			request = BatchEmbeddingRequest(
				model=model_to_use,
				inputs=batch_texts,
				dimensions=dimensions,
				batch_size=batch_size,
				**cast("dict", kwargs),
			)
			try:
				response = await self._provider_client.aembed_batch(
					request, cast("httpx.AsyncClient", self._http_client)
				)
				all_embeddings.extend(response.embeddings)
				# Aggregate usage if present
				if response.usage:
					for k, v in response.usage.items():
						total_usage[k] = total_usage.get(k, 0) + v
				# If provider returns failed_indices, aggregate them
				if response.failed_indices:
					failed_indices.extend([batch_indices[idx] for idx in response.failed_indices])
			except Exception:
				logger.exception(f"Error processing batch starting at {i}")
				failed_indices.extend(batch_indices)
			batch_count += 1
		dimensions = all_embeddings[0].dimensions if all_embeddings else (dimensions or 0)
		return BatchEmbeddingResponse(
			embeddings=all_embeddings,
			dimensions=dimensions,
			texts=texts,
			batch_count=batch_count,
			failed_indices=sorted(set(failed_indices)) if failed_indices else None,
			model=model_to_use,
			provider=self.config.provider,
			usage=total_usage,
		)

	async def close(self) -> None:
		"""Close the underlying HTTP client."""
		if self._http_client and isinstance(self._http_client, _RealHttpxAsyncClient):
			await self._http_client.aclose()
			# self._http_client is set to None by _base_close()
		# Call _base_close() for BaseClient to handle its part (like setting _http_client to None).
		self._base_close()  # Call the renamed method in BaseClient

	async def __aenter__(self) -> "AsyncClient":
		"""Enter async context manager."""
		return self

	async def __aexit__(self, *args: object) -> None:
		"""Exit async context manager and close the client."""
		await self.close()
