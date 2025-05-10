"""
Voyage AI provider client for Embed-All.

This module contains the Voyage-specific client implementation.
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
from embed_all.models.responses.providers.voyage import VoyageAPIResponse

logger = logging.getLogger("embed_all")

# Default model for Voyage AI
DEFAULT_MODEL = "voyage-2"

# Default endpoint URLs
DEFAULT_BASE_URL = "https://api.voyageai.com/v1"
EMBEDDINGS_ENDPOINT = "/embeddings"

# HTTP status codes
HTTP_STATUS_UNAUTHORIZED = 401
HTTP_STATUS_BAD_REQUEST = 400
HTTP_STATUS_RATE_LIMIT = 429

# Maximum batch size for Voyage API
VOYAGE_API_MAX_BATCH_SIZE = 128


class VoyageClient(BaseProviderClient):
	"""Voyage AI client implementation."""

	def __init__(self, config: ProviderModel) -> None:
		"""Initialize the Voyage client.

		Args:
			config: Provider configuration
		"""
		super().__init__(config)
		self.base_url = config.base_url or DEFAULT_BASE_URL

		if not config.api_key:
			msg = "API key is required for Voyage AI"
			raise AuthenticationError(
				msg,
				provider="voyage",
			)

	@property
	def default_model(self) -> str:
		"""Return the default model for Voyage AI."""
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
		parsed_response: VoyageAPIResponse,
		original_texts: list[str],
		request_model_name: str,
	) -> TextEmbeddingResponse:
		"""Process the response from the embeddings API.

		Args:
			parsed_response: The validated Voyage AI API response model.
			original_texts: The original texts used for the request.
			request_model_name: The model name used in the original request.

		Returns:
			TextEmbeddingResponse containing the embeddings.
		"""
		data = parsed_response.data
		if not data:
			return TextEmbeddingResponse(
				model=request_model_name,
				provider=self.config.provider,
				embeddings=[],
				dimensions=0,
				texts=original_texts,
				usage=parsed_response.usage.model_dump() if parsed_response.usage else None,
			)

		embeddings = []
		for item in data:
			embedding_values = [float(val) for val in item.embedding]
			idx = item.index
			text_content = original_texts[idx] if idx < len(original_texts) else None
			embedding = EmbeddingModel(
				values=embedding_values,
				dimensions=len(embedding_values),
				index=idx,
				text=text_content,
				model=request_model_name,
			)
			embeddings.append(embedding)

		embeddings.sort(key=lambda emb: emb.index)
		dimensions = embeddings[0].dimensions if embeddings else 0

		return TextEmbeddingResponse(
			embeddings=embeddings,
			dimensions=dimensions,
			texts=original_texts,
			model=parsed_response.model,
			provider=self.config.provider,
			usage={"total_tokens": parsed_response.usage.total_tokens},
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
		if request.input_type:
			payload["input_type"] = request.input_type
		if request.truncate is not None:
			payload["truncation"] = request.truncate
		if request.dimensions is not None:
			payload["output_dimension"] = request.dimensions
		if request.encoding_format is not None:
			payload["encoding_format"] = request.encoding_format

		original_texts = [request.input] if isinstance(request.input, str) else request.input

		try:
			response = client.post(url, headers=headers, json=payload)
			response.raise_for_status()
			parsed_response = VoyageAPIResponse.model_validate(response.json())
			return self._process_embeddings_response(parsed_response, original_texts, request.model)
		except httpx.HTTPStatusError as e:
			logger.exception("HTTP error")
			response_json = {}
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
				msg = response_json.get("detail", e.response.text)
				raise InvalidRequestError(
					msg,
					provider=self.config.provider,
					status_code=e.response.status_code,
					response=response_json,
				) from e

			api_error_msg = response_json.get("detail", e.response.text)
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
		"""Generate embeddings for a batch of texts.

		Voyage AI has a max batch size of 128.
		This method handles batching by splitting larger requests.
		"""
		all_embeddings_models: list[EmbeddingModel] = []
		failed_indices: list[int] = []
		total_tokens = 0

		original_input_texts = request.inputs
		num_texts = len(original_input_texts)
		url = f"{self.base_url}{EMBEDDINGS_ENDPOINT}"
		headers = self._get_headers()

		for i in range(0, num_texts, VOYAGE_API_MAX_BATCH_SIZE):
			batch_texts = original_input_texts[i : i + VOYAGE_API_MAX_BATCH_SIZE]
			global_start_index = i

			batch_payload: dict[str, Any] = {
				"model": request.model,
				"input": batch_texts,
			}
			if request.input_type:
				batch_payload["input_type"] = request.input_type
			if request.truncate is not None:
				batch_payload["truncation"] = request.truncate
			if request.dimensions is not None:
				batch_payload["output_dimension"] = request.dimensions
			if request.encoding_format is not None:
				batch_payload["encoding_format"] = request.encoding_format

			try:
				api_response = client.post(url, headers=headers, json=batch_payload)
				api_response.raise_for_status()
				parsed_batch_response = VoyageAPIResponse.model_validate(api_response.json())

				batch_data = parsed_batch_response.data
				if not batch_data:
					logger.warning(f"No embedding data in API response for batch starting at {global_start_index}")
					failed_indices.extend(range(global_start_index, global_start_index + len(batch_texts)))
					continue

				for item in batch_data:
					embedding_values = [float(val) for val in item.embedding]
					# Voyage index is 0-based for the current batch
					item_global_idx = global_start_index + item.index

					text_for_embedding = None
					if item_global_idx < len(original_input_texts):
						text_for_embedding = original_input_texts[item_global_idx]

					emb_model = EmbeddingModel(
						values=embedding_values,
						dimensions=len(embedding_values),
						index=item_global_idx,
						text=text_for_embedding,
						model=parsed_batch_response.model,  # Use model from parsed API response
					)
					all_embeddings_models.append(emb_model)

				if parsed_batch_response.usage:
					total_tokens += parsed_batch_response.usage.total_tokens

			except httpx.HTTPStatusError as e:
				logger.exception(f"HTTP error for batch starting at {global_start_index}")
				failed_indices.extend(range(global_start_index, global_start_index + len(batch_texts)))
				# Specific error check for "Too many inputs"
				with contextlib.suppress(json.JSONDecodeError, httpx.DecodingError):
					response_json = e.response.json()
					if response_json.get("detail") == "Too many inputs, max 128":
						logger.warning(f"Batch size error (Too many inputs) for batch starting at {global_start_index}")
			except Exception:
				logger.exception(f"Error processing batch starting at {global_start_index}")
				failed_indices.extend(range(global_start_index, global_start_index + len(batch_texts)))

		# Sort by original index and determine overall dimensions
		all_embeddings_models.sort(key=lambda emb: cast("int", emb.index))
		final_dimensions = all_embeddings_models[0].dimensions if all_embeddings_models else (request.dimensions or 0)

		return BatchEmbeddingResponse(
			embeddings=all_embeddings_models,
			dimensions=final_dimensions,
			texts=original_input_texts,
			batch_count=(num_texts + VOYAGE_API_MAX_BATCH_SIZE - 1) // VOYAGE_API_MAX_BATCH_SIZE,
			failed_indices=cast("list[int]", sorted(set(failed_indices))) if failed_indices else None,
			model=request.model,  # Overall model from original request
			provider=self.config.provider,
			usage={"total_tokens": total_tokens},
		)

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
		if request.input_type:
			payload["input_type"] = request.input_type
		if request.truncate is not None:
			payload["truncation"] = request.truncate
		if request.dimensions is not None:
			payload["output_dimension"] = request.dimensions
		if request.encoding_format is not None:
			payload["encoding_format"] = request.encoding_format

		original_texts = [request.input] if isinstance(request.input, str) else request.input

		try:
			response = await client.post(url, headers=headers, json=payload)
			response.raise_for_status()
			parsed_response = VoyageAPIResponse.model_validate(response.json())
			return self._process_embeddings_response(parsed_response, original_texts, request.model)
		except httpx.HTTPStatusError as e:
			logger.exception("HTTP error")
			response_json = {}
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
				msg = response_json.get("detail", e.response.text)
				raise InvalidRequestError(
					msg,
					provider=self.config.provider,
					status_code=e.response.status_code,
					response=response_json,
				) from e

			api_error_msg = response_json.get("detail", e.response.text)
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

	async def aembed_batch(self, request: BatchEmbeddingRequest, client: httpx.AsyncClient) -> BatchEmbeddingResponse:
		"""Asynchronously generate embeddings for a batch of texts.

		Voyage AI has a max batch size of 128.
		This method handles batching by splitting larger requests.
		"""
		all_embeddings_models: list[EmbeddingModel] = []
		failed_indices: list[int] = []
		total_tokens = 0

		original_input_texts = request.inputs
		num_texts = len(original_input_texts)
		url = f"{self.base_url}{EMBEDDINGS_ENDPOINT}"
		headers = self._get_headers()

		for i in range(0, num_texts, VOYAGE_API_MAX_BATCH_SIZE):
			batch_texts = original_input_texts[i : i + VOYAGE_API_MAX_BATCH_SIZE]
			global_start_index = i

			batch_payload: dict[str, Any] = {
				"model": request.model,
				"input": batch_texts,
			}
			if request.input_type:
				batch_payload["input_type"] = request.input_type
			if request.truncate is not None:
				batch_payload["truncation"] = request.truncate
			if request.dimensions is not None:
				batch_payload["output_dimension"] = request.dimensions
			if request.encoding_format is not None:
				batch_payload["encoding_format"] = request.encoding_format

			try:
				api_response = await client.post(url, headers=headers, json=batch_payload)
				api_response.raise_for_status()
				parsed_batch_response = VoyageAPIResponse.model_validate(api_response.json())

				batch_data = parsed_batch_response.data
				if not batch_data:
					logger.warning(
						f"No embedding data in API response for async batch starting at {global_start_index}"
					)
					failed_indices.extend(range(global_start_index, global_start_index + len(batch_texts)))
					continue

				for item in batch_data:
					embedding_values = [float(val) for val in item.embedding]
					item_global_idx = global_start_index + item.index

					text_for_embedding = None
					if item_global_idx < len(original_input_texts):
						text_for_embedding = original_input_texts[item_global_idx]

					emb_model = EmbeddingModel(
						values=embedding_values,
						dimensions=len(embedding_values),
						index=item_global_idx,
						text=text_for_embedding,
						model=parsed_batch_response.model,  # Use model from parsed API response
					)
					all_embeddings_models.append(emb_model)

				if parsed_batch_response.usage:
					total_tokens += parsed_batch_response.usage.total_tokens

			except httpx.HTTPStatusError as e:
				logger.exception(f"HTTP error for async batch starting at {global_start_index}")
				failed_indices.extend(range(global_start_index, global_start_index + len(batch_texts)))
				with contextlib.suppress(json.JSONDecodeError, httpx.DecodingError):
					response_json = e.response.json()
					if response_json.get("detail") == "Too many inputs, max 128":
						logger.warning(
							f"Batch size error (Too many inputs) for async batch starting at {global_start_index}"
						)
			except Exception:
				logger.exception(f"Error processing async batch starting at {global_start_index}")
				failed_indices.extend(range(global_start_index, global_start_index + len(batch_texts)))

		all_embeddings_models.sort(key=lambda emb: cast("int", emb.index))
		final_dimensions = all_embeddings_models[0].dimensions if all_embeddings_models else (request.dimensions or 0)

		return BatchEmbeddingResponse(
			embeddings=all_embeddings_models,
			dimensions=final_dimensions,
			texts=original_input_texts,
			batch_count=(num_texts + VOYAGE_API_MAX_BATCH_SIZE - 1) // VOYAGE_API_MAX_BATCH_SIZE,
			failed_indices=cast("list[int]", sorted(set(failed_indices))) if failed_indices else None,
			model=request.model,
			provider=self.config.provider,
			usage={"total_tokens": total_tokens},
		)
