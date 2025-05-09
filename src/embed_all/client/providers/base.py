"""
Base provider client for Embed-All.

This module contains the abstract base class for provider-specific clients.
"""

from abc import ABC, abstractmethod
from typing import TypeVar

import httpx

from embed_all.models import (
	BatchEmbeddingRequest,
	BatchEmbeddingResponse,
	ProviderModel,
	TextEmbeddingRequest,
	TextEmbeddingResponse,
)

# Type variable for the client
T = TypeVar("T", bound="BaseProviderClient")


class BaseProviderClient(ABC):
	"""Abstract base class for provider-specific clients.

	This class defines the interface that all provider clients must implement.
	"""

	def __init__(self, config: ProviderModel) -> None:
		"""Initialize the provider client.

		Args:
			config: Provider configuration
		"""
		self.config = config

	@property
	@abstractmethod
	def default_model(self) -> str:
		"""Return the default model for this provider."""

	@abstractmethod
	def embed(self, request: TextEmbeddingRequest, client: httpx.Client) -> TextEmbeddingResponse:
		"""Generate embeddings for text input.

		Args:
			request: The embedding request
			client: The HTTP client to use for the request

		Returns:
			TextEmbeddingResponse containing the embeddings
		"""

	@abstractmethod
	def embed_batch(self, request: BatchEmbeddingRequest, client: httpx.Client) -> BatchEmbeddingResponse:
		"""Generate embeddings for a batch of texts.

		Args:
			request: The batch embedding request
			client: The HTTP client to use for the request

		Returns:
			BatchEmbeddingResponse containing the embeddings
		"""

	@abstractmethod
	async def aembed(self, request: TextEmbeddingRequest, client: httpx.AsyncClient) -> TextEmbeddingResponse:
		"""Asynchronously generate embeddings for text input.

		Args:
			request: The embedding request
			client: The HTTP client to use for the request

		Returns:
			TextEmbeddingResponse containing the embeddings
		"""

	@abstractmethod
	async def aembed_batch(self, request: BatchEmbeddingRequest, client: httpx.AsyncClient) -> BatchEmbeddingResponse:
		"""Asynchronously generate embeddings for a batch of texts.

		Args:
			request: The batch embedding request
			client: The HTTP client to use for the request

		Returns:
			BatchEmbeddingResponse containing the embeddings
		"""
