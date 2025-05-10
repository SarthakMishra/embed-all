"""Ollama provider client implementation."""

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
from embed_all.models.responses.providers.ollama import OllamaEmbeddingResponse

from .base import BaseProviderClient

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "all-minilm"
DEFAULT_BASE_URL = "http://localhost:11434"
EMBEDDINGS_ENDPOINT = "/api/embed"
HTTP_STATUS_OK = 200
HTTP_STATUS_BAD_REQUEST = 400
HTTP_STATUS_UNAUTHORIZED = 401
HTTP_STATUS_RATE_LIMIT = 429
OLLAMA_API_MAX_BATCH_SIZE = 64  # Arbitrary, Ollama supports lists but no explicit limit in docs


class OllamaClient(BaseProviderClient):
	"""Ollama client implementation."""

	def __init__(self, config: ProviderModel) -> None:
		"""Initialize the Ollama client with the given provider configuration."""
		super().__init__(config)
		self.base_url = config.base_url or DEFAULT_BASE_URL

	@property
	def default_model(self) -> str:
		"""Return the default model name for Ollama."""
		return DEFAULT_MODEL

	def _get_headers(self) -> dict[str, str]:
		return {"Content-Type": "application/json"}

	def _process_embeddings_response(
		self,
		parsed_response: OllamaEmbeddingResponse,
		original_texts: list[str],
		request_model_name: str,
	) -> TextEmbeddingResponse:
		"""Convert Ollama API response to standard TextEmbeddingResponse."""
		embeddings = [
			EmbeddingModel(
				values=embedding,
				index=i,
				dimensions=len(embedding),
				text=original_texts[i] if i < len(original_texts) else None,
				model=request_model_name,
			)
			for i, embedding in enumerate(parsed_response.embeddings)
		]
		dimensions = len(parsed_response.embeddings[0]) if parsed_response.embeddings else 0
		usage = {}
		if parsed_response.prompt_eval_count is not None:
			usage["prompt_eval_count"] = parsed_response.prompt_eval_count
		if parsed_response.total_duration is not None:
			usage["total_duration"] = parsed_response.total_duration
		if parsed_response.load_duration is not None:
			usage["load_duration"] = parsed_response.load_duration
		return TextEmbeddingResponse(
			provider="ollama",
			model=request_model_name,
			embeddings=embeddings,
			texts=original_texts,
			dimensions=dimensions,
			usage=usage or None,
		)

	def embed(self, request: TextEmbeddingRequest, client: httpx.Client) -> TextEmbeddingResponse:
		"""Generate embeddings for a single text or list of texts using Ollama."""
		url = f"{self.base_url}{EMBEDDINGS_ENDPOINT}"
		headers = self._get_headers()
		payload: dict[str, Any] = {
			"model": request.model,
			"input": [request.input] if isinstance(request.input, str) else request.input,
		}
		if request.truncate is not None:
			payload["truncate"] = bool(request.truncate)
		try:
			response = client.post(url, headers=headers, json=payload)
			if response.status_code != HTTP_STATUS_OK:
				if response.status_code == HTTP_STATUS_UNAUTHORIZED:
					msg = "Authentication failed."
					raise AuthenticationError(msg, provider="ollama")
				if response.status_code == HTTP_STATUS_BAD_REQUEST:
					msg = f"Invalid request: {response.text}"
					raise InvalidRequestError(msg, provider="ollama")
				if response.status_code == HTTP_STATUS_RATE_LIMIT:
					msg = "Rate limit exceeded."
					raise RateLimitError(msg, provider="ollama")
				msg = f"API request failed with status {response.status_code}: {response.text}"
				raise APIError(msg, provider="ollama")
			response_json = response.json()
			parsed_response = OllamaEmbeddingResponse.model_validate(response_json)
			original_texts = [request.input] if isinstance(request.input, str) else request.input
			return self._process_embeddings_response(parsed_response, original_texts, request.model)
		except httpx.HTTPError as e:
			msg = f"HTTP error occurred: {e!s}"
			raise APIError(msg, provider="ollama") from e
		except json.JSONDecodeError as e:
			msg = f"Failed to parse API response: {e!s}"
			raise APIError(msg, provider="ollama") from e
		except Exception as e:
			msg = f"Unexpected error: {e!s}"
			raise APIError(msg, provider="ollama") from e

	async def aembed(self, request: TextEmbeddingRequest, client: httpx.AsyncClient) -> TextEmbeddingResponse:
		"""Asynchronously generate embeddings for a single text or list of texts using Ollama."""
		url = f"{self.base_url}{EMBEDDINGS_ENDPOINT}"
		headers = self._get_headers()
		payload: dict[str, Any] = {
			"model": request.model,
			"input": [request.input] if isinstance(request.input, str) else request.input,
		}
		if request.truncate is not None:
			payload["truncate"] = bool(request.truncate)
		try:
			response = await client.post(url, headers=headers, json=payload)
			if response.status_code != HTTP_STATUS_OK:
				if response.status_code == HTTP_STATUS_UNAUTHORIZED:
					msg = "Authentication failed."
					raise AuthenticationError(msg, provider="ollama")
				if response.status_code == HTTP_STATUS_BAD_REQUEST:
					msg = f"Invalid request: {response.text}"
					raise InvalidRequestError(msg, provider="ollama")
				if response.status_code == HTTP_STATUS_RATE_LIMIT:
					msg = "Rate limit exceeded."
					raise RateLimitError(msg, provider="ollama")
				msg = f"API request failed with status {response.status_code}: {response.text}"
				raise APIError(msg, provider="ollama")
			response_json = response.json()
			parsed_response = OllamaEmbeddingResponse.model_validate(response_json)
			original_texts = [request.input] if isinstance(request.input, str) else request.input
			return self._process_embeddings_response(parsed_response, original_texts, request.model)
		except httpx.HTTPError as e:
			msg = f"HTTP error occurred: {e!s}"
			raise APIError(msg, provider="ollama") from e
		except json.JSONDecodeError as e:
			msg = f"Failed to parse API response: {e!s}"
			raise APIError(msg, provider="ollama") from e
		except Exception as e:
			msg = f"Unexpected error: {e!s}"
			raise APIError(msg, provider="ollama") from e

	def embed_batch(self, request: BatchEmbeddingRequest, client: httpx.Client) -> BatchEmbeddingResponse:
		"""Generate embeddings for a batch of texts using Ollama."""
		all_embeddings_models = []
		failed_indices: list[int] = []
		original_input_texts = request.inputs
		num_texts = len(original_input_texts)
		url = f"{self.base_url}{EMBEDDINGS_ENDPOINT}"
		headers = self._get_headers()
		for i in range(0, num_texts, min(request.batch_size, OLLAMA_API_MAX_BATCH_SIZE)):
			batch_texts = request.inputs[i : i + min(request.batch_size, OLLAMA_API_MAX_BATCH_SIZE)]
			payload: dict[str, Any] = {
				"model": request.model,
				"input": batch_texts,
			}
			if request.truncate is not None:
				payload["truncate"] = bool(request.truncate)
			try:
				response = client.post(url, headers=headers, json=payload)
				if response.status_code != HTTP_STATUS_OK:
					failed_indices.extend(range(i, min(i + len(batch_texts), num_texts)))
					logger.error(
						"Batch %d-%d failed with status %d: %s",
						i,
						i + len(batch_texts) - 1,
						response.status_code,
						response.text,
					)
					continue
				response_json = response.json()
				parsed_response = OllamaEmbeddingResponse.model_validate(response_json)
				for j, emb in enumerate(parsed_response.embeddings):
					global_idx = i + j
					embedding_model = EmbeddingModel(
						values=emb,
						index=global_idx,
						dimensions=len(emb),
						text=original_input_texts[global_idx],
						model=request.model,
					)
					all_embeddings_models.append(embedding_model)
			except Exception:
				failed_indices.extend(range(i, min(i + len(batch_texts), num_texts)))
				logger.exception("Batch %d-%d failed with error", i, i + len(batch_texts) - 1)
				continue
		all_embeddings_models.sort(key=lambda emb: emb.index)
		dimensions = all_embeddings_models[0].dimensions if all_embeddings_models else request.dimensions or 0
		return BatchEmbeddingResponse(
			provider="ollama",
			model=request.model,
			embeddings=all_embeddings_models,
			texts=original_input_texts,
			dimensions=dimensions,
			failed_indices=failed_indices if failed_indices else None,
			batch_count=(num_texts + OLLAMA_API_MAX_BATCH_SIZE - 1) // OLLAMA_API_MAX_BATCH_SIZE,
			usage=None,
		)

	async def aembed_batch(self, request: BatchEmbeddingRequest, client: httpx.AsyncClient) -> BatchEmbeddingResponse:
		"""Asynchronously generate embeddings for a batch of texts using Ollama."""
		all_embeddings_models = []
		failed_indices: list[int] = []
		original_input_texts = request.inputs
		num_texts = len(original_input_texts)
		url = f"{self.base_url}{EMBEDDINGS_ENDPOINT}"
		headers = self._get_headers()
		tasks = []
		for i in range(0, num_texts, min(request.batch_size, OLLAMA_API_MAX_BATCH_SIZE)):
			batch_texts = request.inputs[i : i + min(request.batch_size, OLLAMA_API_MAX_BATCH_SIZE)]
			payload: dict[str, Any] = {
				"model": request.model,
				"input": batch_texts,
			}
			if request.truncate is not None:
				payload["truncate"] = bool(request.truncate)
			batch_info = {"start_idx": i, "size": len(batch_texts)}
			tasks.append((client.post(url, headers=headers, json=payload), batch_info))
		for task, batch_info in tasks:
			start_idx = batch_info["start_idx"]
			size = batch_info["size"]
			try:
				response = await task
				if response.status_code != HTTP_STATUS_OK:
					failed_indices.extend(range(start_idx, start_idx + size))
					logger.error(
						"Batch %d-%d failed with status %d: %s",
						start_idx,
						start_idx + size - 1,
						response.status_code,
						response.text,
					)
					continue
				response_json = response.json()
				parsed_response = OllamaEmbeddingResponse.model_validate(response_json)
				for j, emb in enumerate(parsed_response.embeddings):
					global_idx = start_idx + j
					embedding_model = EmbeddingModel(
						values=emb,
						index=global_idx,
						dimensions=len(emb),
						text=original_input_texts[global_idx],
						model=request.model,
					)
					all_embeddings_models.append(embedding_model)
			except Exception:
				failed_indices.extend(range(start_idx, start_idx + size))
				logger.exception("Batch %d-%d failed with error", start_idx, start_idx + size - 1)
				continue
		all_embeddings_models.sort(key=lambda emb: emb.index)
		dimensions = all_embeddings_models[0].dimensions if all_embeddings_models else request.dimensions or 0
		return BatchEmbeddingResponse(
			provider="ollama",
			model=request.model,
			embeddings=all_embeddings_models,
			texts=original_input_texts,
			dimensions=dimensions,
			failed_indices=failed_indices if failed_indices else None,
			batch_count=(num_texts + OLLAMA_API_MAX_BATCH_SIZE - 1) // OLLAMA_API_MAX_BATCH_SIZE,
			usage=None,
		)
