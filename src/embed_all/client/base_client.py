"""
Base client for Embed-All.

This module contains the abstract base client class used for all embedding providers.
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Literal, NotRequired, TypedDict

import httpx

from embed_all.models import (
	BatchEmbeddingResponse,
	ProviderModel,
	TextEmbeddingResponse,
)
from embed_all.models.errors import (
	APIError,
	AuthenticationError,
	InvalidRequestError,
	RateLimitError,
)

logger = logging.getLogger("embed_all")

# HTTP status codes
HTTP_STATUS_BAD_REQUEST = 400
HTTP_STATUS_UNAUTHORIZED = 401
HTTP_STATUS_RATE_LIMIT = 429


class EmbeddingKwargs(TypedDict, total=False):
	"""TypedDict for embedding function kwargs."""

	encoding_format: NotRequired[Literal["float", "base64"] | None]
	truncate: NotRequired[Literal["NONE", "START", "END"] | None]


logger = logging.getLogger("embed_all")


class BaseClient(ABC):
	"""Abstract base client for embedding operations.

	This class defines the interface that all embedding clients must implement.
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
		"""Initialize the client.

		Args:
			provider: The provider name
			api_key: API key for the provider
			base_url: Base URL for API requests
			model: Default model to use for embeddings
			timeout: Request timeout in seconds
			max_retries: Maximum number of retries for failed requests
		"""
		self.config = ProviderModel(
			provider=provider,
			api_key=api_key,
			base_url=base_url,
			model=model or self._get_default_model(),
			timeout=timeout,
			max_retries=max_retries,
		)
		logger.debug(f"Initialized {provider} client with model {self.config.model}")

	@abstractmethod
	def _get_default_model(self) -> str:
		"""Return the default model for this provider."""

	@abstractmethod
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

	@abstractmethod
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

	@abstractmethod
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

	@abstractmethod
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

	def _handle_error_response(self, response: httpx.Response, message: str | None = None) -> None:
		"""Handle error responses from the API.

		Args:
			response: The HTTP response
			message: Optional error message to include

		Raises:
			AuthenticationError: If authentication failed
			RateLimitError: If rate limit was exceeded
			InvalidRequestError: If the request was invalid
			APIError: For other API errors
		"""
		status_code = response.status_code
		error_data = {}

		try:
			error_data = response.json()
		except (json.JSONDecodeError, httpx.DecodingError):
			error_data = {"error": {"message": response.text}}

		error_message = message or error_data.get("error", {}).get("message", "Unknown error")

		if status_code == HTTP_STATUS_UNAUTHORIZED:
			msg = f"Authentication failed: {error_message}"
			raise AuthenticationError(
				msg,
				provider=self.config.provider,
				status_code=status_code,
				response=error_data,
			)
		if status_code == HTTP_STATUS_RATE_LIMIT:
			retry_after = response.headers.get("retry-after")
			retry_after_seconds = int(retry_after) if retry_after and retry_after.isdigit() else None

			msg = f"Rate limit exceeded: {error_message}"
			raise RateLimitError(
				msg,
				provider=self.config.provider,
				status_code=status_code,
				response=error_data,
				retry_after=retry_after_seconds,
			)
		if status_code == HTTP_STATUS_BAD_REQUEST:
			msg = f"Invalid request: {error_message}"
			raise InvalidRequestError(
				msg,
				provider=self.config.provider,
				status_code=status_code,
				response=error_data,
			)
		msg = f"API error: {error_message}"
		raise APIError(
			msg,
			provider=self.config.provider,
			status_code=status_code,
			response=error_data,
		)
