"""Cohere provider client implementation."""

import json
import logging
from typing import Any

import httpx

from embed_all.models import (
	BatchEmbeddingRequest,
	BatchEmbeddingResponse,
	EmbeddingModel,
	ProviderModel,
	TextEmbeddingRequest,
	TextEmbeddingResponse,
)
from embed_all.models.errors import APIError, AuthenticationError, InvalidRequestError, RateLimitError
from embed_all.models.responses.providers.cohere import CohereAPIResponse

from .base import BaseProviderClient

logger = logging.getLogger(__name__)

# Default model for Cohere
DEFAULT_MODEL = "embed-v4.0"

# Default endpoint URLs
DEFAULT_BASE_URL = "https://api.cohere.com"
EMBEDDINGS_ENDPOINT = "/v2/embed"

# HTTP status codes
HTTP_STATUS_UNAUTHORIZED = 401
HTTP_STATUS_BAD_REQUEST = 400
HTTP_STATUS_RATE_LIMIT = 429
HTTP_STATUS_OK = 200

# Maximum batch size for Cohere API
COHERE_API_MAX_BATCH_SIZE = 96  # Per documentation


class CohereClient(BaseProviderClient):
	"""Cohere client implementation."""

	def __init__(self, config: ProviderModel) -> None:
		"""Initialize the Cohere client.

		Args:
		    config: Provider configuration
		"""
		super().__init__(config)
		# Use the provided base_url or default
		self.base_url = config.base_url or DEFAULT_BASE_URL

	@property
	def default_model(self) -> str:
		"""Return the default model for Cohere."""
		return DEFAULT_MODEL

	def _get_headers(self) -> dict[str, str]:
		"""Get the headers for API requests.

		Returns:
		    Dict with the headers
		"""
		return {
			"Authorization": f"bearer {self.config.api_key}",
			"Content-Type": "application/json",
			"Accept": "application/json",
		}

	def _process_embeddings_response(
		self,
		parsed_response: CohereAPIResponse,
		original_texts: list[str],
		request_model_name: str,
	) -> TextEmbeddingResponse:
		"""Process the response from the embeddings API.

		Args:
		    parsed_response: The validated Cohere API response model.
		    original_texts: The original texts used for the request.
		    request_model_name: The model name used in the original request.

		Returns:
		    TextEmbeddingResponse containing the embeddings.
		"""
		# Extract embeddings from the response
		if "float" not in parsed_response.embeddings:
			msg = "No float embeddings found in response"
			raise APIError(msg, provider=self.config.provider)

		raw_embeddings = parsed_response.embeddings["float"]

		# Convert to list of EmbeddingModel
		embeddings = [
			EmbeddingModel(
				values=embedding, index=i, dimensions=len(embedding), text=original_texts[i], model=request_model_name
			)
			for i, embedding in enumerate(raw_embeddings)
		]

		# Determine dimensions from the first embedding
		dimensions = len(raw_embeddings[0]) if raw_embeddings and raw_embeddings[0] else 0

		# Calculate token usage if available
		tokens_used = None
		if parsed_response.meta.billed_units and "input_tokens" in parsed_response.meta.billed_units:
			tokens_used = parsed_response.meta.billed_units["input_tokens"]

		return TextEmbeddingResponse(
			provider=self.config.provider,
			model=request_model_name,
			embeddings=embeddings,
			texts=original_texts,
			dimensions=dimensions,
			usage={"tokens": tokens_used} if tokens_used is not None else None,
		)

	def embed(self, request: TextEmbeddingRequest, client: httpx.Client) -> TextEmbeddingResponse:
		"""Generate embeddings for text input.

		Args:
		    request: The embedding request
		    client: The HTTP client to use for the request

		Returns:
		    TextEmbeddingResponse containing the embeddings
		"""
		url = f"{self.base_url}{EMBEDDINGS_ENDPOINT}"
		headers = self._get_headers()

		payload: dict[str, Any] = {
			"model": request.model,
			"input_type": "classification",  # Default to classification
			"embedding_types": ["float"],
		}

		# Handle text input - can be a string or list of strings
		original_texts: list[str]
		if isinstance(request.input, str):
			payload["texts"] = [request.input]
			original_texts = [request.input]
		else:
			payload["texts"] = request.input
			original_texts = request.input

		# Add dimensions if specified
		if request.dimensions:
			payload["output_dimension"] = request.dimensions

		# Add truncation parameter if specified
		if request.truncate is not None:
			if request.truncate:
				payload["truncate"] = "END"
			else:
				payload["truncate"] = "NONE"

		# Add input_type if specified
		if request.input_type:
			payload["input_type"] = request.input_type

		try:
			response = client.post(url, headers=headers, json=payload)

			# Handle error responses
			if response.status_code != HTTP_STATUS_OK:
				if response.status_code == HTTP_STATUS_UNAUTHORIZED:
					msg = "Authentication failed. Please check your API key."
					raise AuthenticationError(msg, provider=self.config.provider)
				if response.status_code == HTTP_STATUS_BAD_REQUEST:
					msg = f"Invalid request: {response.text}"
					raise InvalidRequestError(msg, provider=self.config.provider)
				if response.status_code == HTTP_STATUS_RATE_LIMIT:
					msg = "Rate limit exceeded."
					raise RateLimitError(msg, provider=self.config.provider)
				msg = f"API request failed with status {response.status_code}: {response.text}"
				raise APIError(msg, provider=self.config.provider)

			# Parse the response
			response_json = response.json()

			# Validate with Pydantic model
			parsed_response = CohereAPIResponse.parse_obj(response_json)

			# Process into standard format
			return self._process_embeddings_response(parsed_response, original_texts, request.model)

		except httpx.HTTPError as e:
			msg = f"HTTP error occurred: {e!s}"
			raise APIError(msg, provider=self.config.provider) from e

		except json.JSONDecodeError as e:
			msg = f"Failed to parse API response: {e!s}"
			raise APIError(msg, provider=self.config.provider) from e

		except Exception as e:
			msg = f"Unexpected error: {e!s}"
			raise APIError(msg, provider=self.config.provider) from e

	async def aembed(self, request: TextEmbeddingRequest, client: httpx.AsyncClient) -> TextEmbeddingResponse:
		"""Asynchronously generate embeddings for text input.

		Args:
		    request: The embedding request
		    client: The HTTP client to use for the request

		Returns:
		    TextEmbeddingResponse containing the embeddings
		"""
		url = f"{self.base_url}{EMBEDDINGS_ENDPOINT}"
		headers = self._get_headers()

		payload: dict[str, Any] = {
			"model": request.model,
			"input_type": "classification",  # Default to classification
			"embedding_types": ["float"],
		}

		# Handle text input - can be a string or list of strings
		original_texts: list[str]
		if isinstance(request.input, str):
			payload["texts"] = [request.input]
			original_texts = [request.input]
		else:
			payload["texts"] = request.input
			original_texts = request.input

		# Add dimensions if specified
		if request.dimensions:
			payload["output_dimension"] = request.dimensions

		# Add truncation parameter if specified
		if request.truncate is not None:
			if request.truncate:
				payload["truncate"] = "END"
			else:
				payload["truncate"] = "NONE"

		# Add input_type if specified
		if request.input_type:
			payload["input_type"] = request.input_type

		try:
			response = await client.post(url, headers=headers, json=payload)

			# Handle error responses
			if response.status_code != HTTP_STATUS_OK:
				if response.status_code == HTTP_STATUS_UNAUTHORIZED:
					msg = "Authentication failed. Please check your API key."
					raise AuthenticationError(msg, provider=self.config.provider)
				if response.status_code == HTTP_STATUS_BAD_REQUEST:
					msg = f"Invalid request: {response.text}"
					raise InvalidRequestError(msg, provider=self.config.provider)
				if response.status_code == HTTP_STATUS_RATE_LIMIT:
					msg = "Rate limit exceeded."
					raise RateLimitError(msg, provider=self.config.provider)
				msg = f"API request failed with status {response.status_code}: {response.text}"
				raise APIError(msg, provider=self.config.provider)

			# Parse the response
			response_json = response.json()

			# Validate with Pydantic model
			parsed_response = CohereAPIResponse.parse_obj(response_json)

			# Process into standard format
			return self._process_embeddings_response(parsed_response, original_texts, request.model)

		except httpx.HTTPError as e:
			msg = f"HTTP error occurred: {e!s}"
			raise APIError(msg, provider=self.config.provider) from e

		except json.JSONDecodeError as e:
			msg = f"Failed to parse API response: {e!s}"
			raise APIError(msg, provider=self.config.provider) from e

		except Exception as e:
			msg = f"Unexpected error: {e!s}"
			raise APIError(msg, provider=self.config.provider) from e

	def embed_batch(self, request: BatchEmbeddingRequest, client: httpx.Client) -> BatchEmbeddingResponse:
		"""Generate embeddings for a batch of texts.

		Args:
		    request: The batch embedding request
		    client: The HTTP client to use for the request

		Returns:
		    BatchEmbeddingResponse containing the embeddings
		"""
		all_embeddings_models = []
		failed_indices: list[int] = []
		total_tokens: int | None = None

		# Keep original texts for returning in response
		original_input_texts = request.inputs
		num_texts = len(original_input_texts)

		# Process in batches according to Cohere's maximum batch size
		url = f"{self.base_url}{EMBEDDINGS_ENDPOINT}"
		headers = self._get_headers()

		for i in range(0, num_texts, min(request.batch_size, COHERE_API_MAX_BATCH_SIZE)):
			batch_texts = request.inputs[i : i + min(request.batch_size, COHERE_API_MAX_BATCH_SIZE)]

			payload: dict[str, Any] = {
				"model": request.model,
				"texts": batch_texts,
				"input_type": "classification",  # Default to classification
				"embedding_types": ["float"],
			}

			# Add dimensions if specified
			if request.dimensions:
				payload["output_dimension"] = request.dimensions

			# Add truncation parameter if specified
			if request.truncate is not None:
				if request.truncate:
					payload["truncate"] = "END"
				else:
					payload["truncate"] = "NONE"

			# Add input_type if specified
			if request.input_type:
				payload["input_type"] = request.input_type

			try:
				response = client.post(url, headers=headers, json=payload)

				# Handle error responses
				if response.status_code != HTTP_STATUS_OK:
					# Add all indices in this batch to failed_indices
					failed_indices.extend(range(i, min(i + len(batch_texts), num_texts)))
					logger.error(
						"Batch %d-%d failed with status %d: %s",
						i,
						i + len(batch_texts) - 1,
						response.status_code,
						response.text,
					)
					continue

				# Parse the response
				response_json = response.json()

				# Validate with Pydantic model
				parsed_response = CohereAPIResponse.parse_obj(response_json)

				# Extract embeddings
				if "float" not in parsed_response.embeddings:
					# If no embeddings, add these indices to failed_indices
					failed_indices.extend(range(i, min(i + len(batch_texts), num_texts)))
					continue

				batch_embeddings = parsed_response.embeddings["float"]

				# Convert to EmbeddingModel instances with correct index
				for j, emb in enumerate(batch_embeddings):
					global_idx = i + j
					embedding_model = EmbeddingModel(
						values=emb,
						index=global_idx,
						dimensions=len(emb),
						text=original_input_texts[global_idx],
						model=request.model,
					)
					all_embeddings_models.append(embedding_model)

				# Track token usage
				if parsed_response.meta.billed_units and "input_tokens" in parsed_response.meta.billed_units:
					batch_tokens = parsed_response.meta.billed_units["input_tokens"]
					total_tokens = (total_tokens or 0) + batch_tokens

			except Exception:
				# Add all indices in this batch to failed_indices
				failed_indices.extend(range(i, min(i + len(batch_texts), num_texts)))
				logger.exception("Batch %d-%d failed with error", i, i + len(batch_texts) - 1)
				continue

		# Sort by original index
		all_embeddings_models.sort(key=lambda emb: emb.index)

		# Determine dimensions from the first embedding
		dimensions = all_embeddings_models[0].dimensions if all_embeddings_models else request.dimensions or 0

		return BatchEmbeddingResponse(
			provider=self.config.provider,
			model=request.model,
			embeddings=all_embeddings_models,
			texts=original_input_texts,
			dimensions=dimensions,
			failed_indices=failed_indices if failed_indices else None,
			batch_count=(num_texts + COHERE_API_MAX_BATCH_SIZE - 1) // COHERE_API_MAX_BATCH_SIZE,
			usage={"tokens": total_tokens} if total_tokens is not None else None,
		)

	async def aembed_batch(self, request: BatchEmbeddingRequest, client: httpx.AsyncClient) -> BatchEmbeddingResponse:
		"""Asynchronously generate embeddings for a batch of texts.

		Args:
		    request: The batch embedding request
		    client: The HTTP client to use for the request

		Returns:
		    BatchEmbeddingResponse containing the embeddings
		"""
		all_embeddings_models = []
		failed_indices: list[int] = []
		total_tokens: int | None = None

		# Keep original texts for returning in response
		original_input_texts = request.inputs
		num_texts = len(original_input_texts)

		# Process in batches according to Cohere's maximum batch size
		url = f"{self.base_url}{EMBEDDINGS_ENDPOINT}"
		headers = self._get_headers()

		# Create and collect all tasks
		tasks = []
		for i in range(0, num_texts, min(request.batch_size, COHERE_API_MAX_BATCH_SIZE)):
			batch_texts = request.inputs[i : i + min(request.batch_size, COHERE_API_MAX_BATCH_SIZE)]

			payload: dict[str, Any] = {
				"model": request.model,
				"texts": batch_texts,
				"input_type": "classification",  # Default to classification
				"embedding_types": ["float"],
			}

			# Add dimensions if specified
			if request.dimensions:
				payload["output_dimension"] = request.dimensions

			# Add truncation parameter if specified
			if request.truncate is not None:
				if request.truncate:
					payload["truncate"] = "END"
				else:
					payload["truncate"] = "NONE"

			# Add input_type if specified
			if request.input_type:
				payload["input_type"] = request.input_type

			# Store batch index with task for processing results
			batch_info = {"start_idx": i, "size": len(batch_texts)}
			task = client.post(url, headers=headers, json=payload)
			tasks.append((task, batch_info))

		# Process all tasks as they complete
		for task, batch_info in tasks:
			start_idx = batch_info["start_idx"]
			size = batch_info["size"]

			try:
				response = await task

				# Handle error responses
				if response.status_code != HTTP_STATUS_OK:
					# Add all indices in this batch to failed_indices
					failed_indices.extend(range(start_idx, start_idx + size))
					logger.error(
						"Batch %d-%d failed with status %d: %s",
						start_idx,
						start_idx + size - 1,
						response.status_code,
						response.text,
					)
					continue

				# Parse the response
				response_json = response.json()

				# Validate with Pydantic model
				parsed_response = CohereAPIResponse.parse_obj(response_json)

				# Extract embeddings
				if "float" not in parsed_response.embeddings:
					# If no embeddings, add these indices to failed_indices
					failed_indices.extend(range(start_idx, start_idx + size))
					continue

				batch_embeddings = parsed_response.embeddings["float"]

				# Convert to EmbeddingModel instances with correct index
				for j, emb in enumerate(batch_embeddings):
					global_idx = start_idx + j
					embedding_model = EmbeddingModel(
						values=emb,
						index=global_idx,
						dimensions=len(emb),
						text=original_input_texts[global_idx],
						model=request.model,
					)
					all_embeddings_models.append(embedding_model)

				# Track token usage
				if parsed_response.meta.billed_units and "input_tokens" in parsed_response.meta.billed_units:
					batch_tokens = parsed_response.meta.billed_units["input_tokens"]
					total_tokens = (total_tokens or 0) + batch_tokens

			except Exception:
				# Add all indices in this batch to failed_indices
				failed_indices.extend(range(start_idx, start_idx + size))
				logger.exception("Batch %d-%d failed with error", start_idx, start_idx + size - 1)
				continue

		# Sort by original index
		all_embeddings_models.sort(key=lambda emb: emb.index)

		# Determine dimensions from the first embedding
		dimensions = all_embeddings_models[0].dimensions if all_embeddings_models else request.dimensions or 0

		return BatchEmbeddingResponse(
			provider=self.config.provider,
			model=request.model,
			embeddings=all_embeddings_models,
			texts=original_input_texts,
			dimensions=dimensions,
			failed_indices=failed_indices if failed_indices else None,
			batch_count=(num_texts + COHERE_API_MAX_BATCH_SIZE - 1) // COHERE_API_MAX_BATCH_SIZE,
			usage={"tokens": total_tokens} if total_tokens is not None else None,
		)
