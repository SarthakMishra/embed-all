"""
OpenAI provider client for Embed-All.

This module contains the OpenAI-specific client implementation.
"""

import contextlib
import json
import logging
from typing import Any, cast

import httpx

from embed_all.client.providers.base import BaseProviderClient
from embed_all.models import (
	BatchEmbeddingRequest,
	BatchEmbeddingResponse,
	EmbeddingModel,
	ProviderModel,
	TextEmbeddingRequest,
	TextEmbeddingResponse,
)
from embed_all.models.errors import (
	APIError,
	AuthenticationError,
	InvalidRequestError,
	RateLimitError,
)
from embed_all.models.responses.providers.open_ai import OpenAIAPIResponse

logger = logging.getLogger("embed_all")

# Default model for OpenAI
DEFAULT_MODEL = "text-embedding-3-small"
OPENAI_API_MAX_BATCH_SIZE = 2048

# Default endpoint URLs
DEFAULT_BASE_URL = "https://api.openai.com/v1"
EMBEDDINGS_ENDPOINT = "/embeddings"

# HTTP status codes
HTTP_STATUS_UNAUTHORIZED = 401
HTTP_STATUS_BAD_REQUEST = 400
HTTP_STATUS_RATE_LIMIT = 429


class OpenAIClient(BaseProviderClient):
	"""OpenAI client implementation."""

	def __init__(self, config: ProviderModel) -> None:
		"""Initialize the OpenAI client.

		Args:
		    config: Provider configuration
		"""
		super().__init__(config)
		self.base_url = config.base_url or DEFAULT_BASE_URL

		if not config.api_key:
			msg = "API key is required for OpenAI"
			raise AuthenticationError(msg, provider="openai")

	@property
	def default_model(self) -> str:
		"""Return the default model for OpenAI."""
		return DEFAULT_MODEL

	def _get_headers(self) -> dict[str, str]:
		"""Get the headers for API requests.

		Returns:
		    Dict with the headers
		"""
		return {
			"Authorization": f"Bearer {self.config.api_key}",
			"Content-Type": "application/json",
		}

	def _process_embeddings_response(
		self,
		parsed_response: OpenAIAPIResponse,
		original_texts: list[str],
	) -> TextEmbeddingResponse:
		"""Process the response from the embeddings API.

		Args:
		    parsed_response: The validated OpenAI API response model
		    original_texts: The original texts used for the request (for context)

		Returns:
		    TextEmbeddingResponse containing the embeddings
		"""
		data = parsed_response.data
		if not data:
			msg = "No embeddings returned from API"
			raise APIError(msg, provider=self.config.provider)

		embeddings = []
		# OpenAI returns embeddings in the order of the input
		for item in data:
			embedding_values = item.embedding
			idx = item.index
			text_content = original_texts[idx] if idx < len(original_texts) else None

			embedding = EmbeddingModel(
				values=embedding_values,
				dimensions=len(embedding_values),
				index=idx,
				text=text_content,
				model=self.config.model,
			)
			embeddings.append(embedding)

		# Sort by index to ensure original order, though OpenAI should preserve it
		embeddings.sort(key=lambda emb: emb.index)

		# Determine overall dimensions from the first embedding (assuming all are same)
		dimensions = embeddings[0].dimensions if embeddings else 0

		# The original_texts for a single API call will correspond to the texts sent in that call.
		# If TextEmbeddingRequest had a single string, original_texts will be a list with one item.
		# If TextEmbeddingRequest had a list of strings, original_texts will be that list.
		# The 'texts' field in TextEmbeddingResponse should reflect the input structure that led to these embeddings.

		return TextEmbeddingResponse(
			embeddings=embeddings,
			dimensions=dimensions,
			texts=original_texts,
			model=parsed_response.model,
			provider=self.config.provider,
			usage={
				"prompt_tokens": parsed_response.usage.prompt_tokens,
				"total_tokens": parsed_response.usage.total_tokens,
			},
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
			"input": request.input,
		}

		if request.dimensions:
			payload["dimensions"] = request.dimensions
		if request.encoding_format:
			payload["encoding_format"] = request.encoding_format
		# OpenAI does not support a 'truncate' parameter directly in the API.
		# It truncates automatically if input exceeds model's context length.

		# The 'input' to OpenAI API can be a string or a list of strings.
		# Our TextEmbeddingRequest.input is str | list[str].
		# We need original_texts for _process_embeddings_response.
		original_texts = [request.input] if isinstance(request.input, str) else request.input

		try:
			response = client.post(url, headers=headers, json=payload)
			response.raise_for_status()
			parsed_response = OpenAIAPIResponse.model_validate(response.json())
			return self._process_embeddings_response(parsed_response, original_texts)
		except httpx.HTTPStatusError as e:
			logger.exception("HTTP error")
			response_json = {}
			# Try to get JSON response, but suppress errors if it's not valid JSON
			with contextlib.suppress(json.JSONDecodeError, httpx.DecodingError):
				response_json = e.response.json()

			if e.response.status_code == HTTP_STATUS_UNAUTHORIZED:
				msg = "Authentication failed"
				raise AuthenticationError(
					msg,
					provider=self.config.provider,
					status_code=e.response.status_code,
					response=response_json,
				) from e
			if e.response.status_code == HTTP_STATUS_RATE_LIMIT:
				msg = "Rate limit exceeded"
				raise RateLimitError(
					msg,
					provider=self.config.provider,
					status_code=e.response.status_code,
					response=response_json,
				) from e
			if e.response.status_code == HTTP_STATUS_BAD_REQUEST:
				msg = response_json.get("error", {}).get("message", e.response.text)
				raise InvalidRequestError(
					msg,
					provider=self.config.provider,
					status_code=e.response.status_code,
					response=response_json,
				) from e

			api_error_msg = response_json.get("error", {}).get("message", e.response.text)
			raise APIError(
				api_error_msg,
				provider=self.config.provider,
				status_code=e.response.status_code,
				response=response_json,
			) from e
		except httpx.RequestError as e:
			logger.exception("Request error")
			msg = f"Request error: {e!s}"
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
			"input": request.input,
		}

		if request.dimensions:
			payload["dimensions"] = request.dimensions
		if request.encoding_format:
			payload["encoding_format"] = request.encoding_format

		original_texts = [request.input] if isinstance(request.input, str) else request.input

		try:
			response = await client.post(url, headers=headers, json=payload)
			response.raise_for_status()
			parsed_response = OpenAIAPIResponse.model_validate(response.json())
			return self._process_embeddings_response(parsed_response, original_texts)
		except httpx.HTTPStatusError as e:
			logger.exception("HTTP error")
			response_json = {}
			# Try to get JSON response, but suppress errors if it's not valid JSON
			with contextlib.suppress(json.JSONDecodeError, httpx.DecodingError):
				response_json = e.response.json()

			if e.response.status_code == HTTP_STATUS_UNAUTHORIZED:
				msg = "Authentication failed"
				raise AuthenticationError(
					msg,
					provider=self.config.provider,
					status_code=e.response.status_code,
					response=response_json,
				) from e
			if e.response.status_code == HTTP_STATUS_RATE_LIMIT:
				msg = "Rate limit exceeded"
				raise RateLimitError(
					msg,
					provider=self.config.provider,
					status_code=e.response.status_code,
					response=response_json,
				) from e
			if e.response.status_code == HTTP_STATUS_BAD_REQUEST:
				msg = response_json.get("error", {}).get("message", e.response.text)
				raise InvalidRequestError(
					msg,
					provider=self.config.provider,
					status_code=e.response.status_code,
					response=response_json,
				) from e

			api_error_msg = response_json.get("error", {}).get("message", e.response.text)
			raise APIError(
				api_error_msg,
				provider=self.config.provider,
				status_code=e.response.status_code,
				response=response_json,
			) from e
		except httpx.RequestError as e:
			logger.exception("Request error")
			msg = f"Request error: {e!s}"
			raise APIError(msg, provider=self.config.provider) from e

	def embed_batch(self, request: BatchEmbeddingRequest, client: httpx.Client) -> BatchEmbeddingResponse:
		"""Generate embeddings for a batch of texts."""
		all_embeddings_models: list[EmbeddingModel] = []
		failed_indices: list[int] = []
		total_usage: dict[str, int] = {}

		original_input_texts = request.inputs
		num_texts = len(original_input_texts)
		url = f"{self.base_url}{EMBEDDINGS_ENDPOINT}"
		headers = self._get_headers()

		for i in range(0, num_texts, OPENAI_API_MAX_BATCH_SIZE):
			batch_texts = original_input_texts[i : i + OPENAI_API_MAX_BATCH_SIZE]
			current_batch_global_start_idx = i

			logger.debug(
				f"Processing batch {i // OPENAI_API_MAX_BATCH_SIZE + 1}/"
				f"{(num_texts + OPENAI_API_MAX_BATCH_SIZE - 1) // OPENAI_API_MAX_BATCH_SIZE}"
				f" for texts starting at global index {current_batch_global_start_idx}"
			)

			payload: dict[str, Any] = {
				"model": request.model,
				"input": batch_texts,
			}
			if request.dimensions:
				payload["dimensions"] = request.dimensions
			if request.encoding_format:
				payload["encoding_format"] = request.encoding_format

			try:
				api_response = client.post(url, headers=headers, json=payload)
				api_response.raise_for_status()
				# Validate and parse the API response for the current batch
				parsed_batch_response = OpenAIAPIResponse.model_validate(api_response.json())

				# Process response for this sub-batch
				data = parsed_batch_response.data
				if not data:
					# If API returns 200 but no data for this batch, mark all as failed for this batch
					logger.warning(
						f"No embedding data in API response for batch at idx {current_batch_global_start_idx}"
					)
					failed_indices.extend(
						range(current_batch_global_start_idx, current_batch_global_start_idx + len(batch_texts))
					)
					continue

				for _item_idx, item in enumerate(data):
					embedding_values = item.embedding
					# OpenAI 'index' in response refers to index within the batch_texts sent
					original_item_idx_in_batch = item.index  # Use validated index
					global_idx = current_batch_global_start_idx + original_item_idx_in_batch

					emb_model = EmbeddingModel(
						values=embedding_values,
						dimensions=len(embedding_values),
						index=global_idx,  # Use global index
						text=original_input_texts[global_idx] if global_idx < num_texts else None,
						model=parsed_batch_response.model,  # Model from parsed API response
					)
					all_embeddings_models.append(emb_model)

				current_usage_model = parsed_batch_response.usage
				if current_usage_model:
					total_usage["prompt_tokens"] = (
						total_usage.get("prompt_tokens", 0) + current_usage_model.prompt_tokens
					)
					total_usage["total_tokens"] = total_usage.get("total_tokens", 0) + current_usage_model.total_tokens

			except httpx.HTTPStatusError:
				logger.exception(f"HTTP error for batch starting at {current_batch_global_start_idx}")
				failed_indices.extend(
					range(current_batch_global_start_idx, current_batch_global_start_idx + len(batch_texts))
				)
				# Optionally, parse e.response.json() if needed for more detailed error logging per batch
			except Exception:  # Keep generic Exception here for broader catch during batch processing
				logger.exception(f"Error processing batch starting at {current_batch_global_start_idx}")
				failed_indices.extend(
					range(current_batch_global_start_idx, current_batch_global_start_idx + len(batch_texts))
				)

		# Ensure sorting by a non-optional integer index
		all_embeddings_models.sort(key=lambda emb: cast("int", emb.index))
		dimensions = all_embeddings_models[0].dimensions if all_embeddings_models else (request.dimensions or 0)

		return BatchEmbeddingResponse(
			embeddings=all_embeddings_models,
			dimensions=dimensions,
			texts=original_input_texts,
			batch_count=(num_texts + OPENAI_API_MAX_BATCH_SIZE - 1) // OPENAI_API_MAX_BATCH_SIZE,
			failed_indices=sorted(set(failed_indices)),
			model=request.model,
			provider=self.config.provider,
			usage=total_usage,
		)

	async def aembed_batch(self, request: BatchEmbeddingRequest, client: httpx.AsyncClient) -> BatchEmbeddingResponse:
		"""Asynchronously generate embeddings for a batch of texts."""
		all_embeddings_models: list[EmbeddingModel] = []
		failed_indices: list[int] = []
		total_usage: dict[str, int] = {}

		original_input_texts = request.inputs
		num_texts = len(original_input_texts)
		url = f"{self.base_url}{EMBEDDINGS_ENDPOINT}"
		headers = self._get_headers()

		for i in range(0, num_texts, OPENAI_API_MAX_BATCH_SIZE):
			batch_texts = original_input_texts[i : i + OPENAI_API_MAX_BATCH_SIZE]
			current_batch_global_start_idx = i

			logger.debug(
				f"Processing async batch {i // OPENAI_API_MAX_BATCH_SIZE + 1}/"
				f"{(num_texts + OPENAI_API_MAX_BATCH_SIZE - 1) // OPENAI_API_MAX_BATCH_SIZE}"
				f" for texts starting at global index {current_batch_global_start_idx}"
			)

			payload: dict[str, Any] = {
				"model": request.model,
				"input": batch_texts,
			}
			if request.dimensions:
				payload["dimensions"] = request.dimensions
			if request.encoding_format:
				payload["encoding_format"] = request.encoding_format

			try:
				api_response = await client.post(url, headers=headers, json=payload)
				api_response.raise_for_status()
				# Validate and parse the API response for the current batch
				parsed_batch_response = OpenAIAPIResponse.model_validate(api_response.json())

				data = parsed_batch_response.data
				if not data:
					logger.warning(
						f"No embedding data in API response for async batch at idx {current_batch_global_start_idx}"
					)
					failed_indices.extend(
						range(current_batch_global_start_idx, current_batch_global_start_idx + len(batch_texts))
					)
					continue

				for _item_idx, item in enumerate(data):
					embedding_values = item.embedding
					original_item_idx_in_batch = item.index  # Use validated index
					global_idx = current_batch_global_start_idx + original_item_idx_in_batch

					emb_model = EmbeddingModel(
						values=embedding_values,
						dimensions=len(embedding_values),
						index=global_idx,
						text=original_input_texts[global_idx] if global_idx < num_texts else None,
						model=parsed_batch_response.model,  # Model from parsed API response
					)
					all_embeddings_models.append(emb_model)

				current_usage_model = parsed_batch_response.usage
				if current_usage_model:
					total_usage["prompt_tokens"] = (
						total_usage.get("prompt_tokens", 0) + current_usage_model.prompt_tokens
					)
					total_usage["total_tokens"] = total_usage.get("total_tokens", 0) + current_usage_model.total_tokens

			except httpx.HTTPStatusError:
				logger.exception(f"HTTP error for async batch starting at {current_batch_global_start_idx}")
				failed_indices.extend(
					range(current_batch_global_start_idx, current_batch_global_start_idx + len(batch_texts))
				)
			except Exception:  # Keep generic Exception here for broader catch during batch processing
				logger.exception(f"Error processing async batch starting at {current_batch_global_start_idx}")
				failed_indices.extend(
					range(current_batch_global_start_idx, current_batch_global_start_idx + len(batch_texts))
				)

		# Ensure sorting by a non-optional integer index
		all_embeddings_models.sort(key=lambda emb: cast("int", emb.index))
		dimensions = all_embeddings_models[0].dimensions if all_embeddings_models else (request.dimensions or 0)

		return BatchEmbeddingResponse(
			embeddings=all_embeddings_models,
			dimensions=dimensions,
			texts=original_input_texts,
			batch_count=(num_texts + OPENAI_API_MAX_BATCH_SIZE - 1) // OPENAI_API_MAX_BATCH_SIZE,
			failed_indices=sorted(set(failed_indices)),
			model=request.model,
			provider=self.config.provider,
			usage=total_usage,
		)
