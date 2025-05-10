"""Tests for src/embed_all/client/providers/ollama.py."""

import httpx
import pytest

from embed_all.client.providers.ollama import OllamaClient
from embed_all.models import (
	BatchEmbeddingRequest,
	BatchEmbeddingResponse,
	ProviderModel,
	TextEmbeddingRequest,
	TextEmbeddingResponse,
)
from embed_all.models.responses.providers.ollama import OllamaEmbeddingResponse


def test_ollama_client_initialization():
	"""Test OllamaClient initialization with and without base_url."""
	client = OllamaClient(
		ProviderModel(
			provider="ollama",
			api_key=None,
			base_url=None,
			model=None,
			timeout=60,
			max_retries=3,
		)
	)
	assert client.base_url == "http://localhost:11434"
	client2 = OllamaClient(
		ProviderModel(
			provider="ollama",
			api_key=None,
			base_url="http://custom:1234",
			model=None,
			timeout=60,
			max_retries=3,
		)
	)
	assert client2.base_url == "http://custom:1234"


def test_ollama_client_get_headers():
	"""Test OllamaClient._get_headers returns the correct headers."""
	client = OllamaClient(
		ProviderModel(
			provider="ollama",
			api_key=None,
			base_url=None,
			model=None,
			timeout=60,
			max_retries=3,
		)
	)
	headers = client._get_headers()
	assert headers["Content-Type"] == "application/json"
	assert len(headers) == 1


def test_ollama_client_process_embeddings_response_valid():
	"""Test OllamaClient._process_embeddings_response with a valid API response."""
	client = OllamaClient(
		ProviderModel(provider="ollama", api_key=None, base_url=None, model=None, timeout=60, max_retries=3)
	)
	request_model_name = "all-minilm"
	input_texts = ["Hello world", "Another text"]
	api_json_response = {
		"model": request_model_name,
		"embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
		"total_duration": 12345,
		"load_duration": 2345,
		"prompt_eval_count": 2,
	}
	validated_api_response = OllamaEmbeddingResponse.model_validate(api_json_response)
	processed_response = client._process_embeddings_response(validated_api_response, input_texts, request_model_name)
	assert isinstance(processed_response, TextEmbeddingResponse)
	assert len(processed_response.embeddings) == 2
	assert processed_response.texts == input_texts
	assert processed_response.model == request_model_name
	assert processed_response.provider == "ollama"
	assert processed_response.usage == {
		"prompt_eval_count": 2,
		"total_duration": 12345,
		"load_duration": 2345,
	}
	assert processed_response.embeddings[0].values == [0.1, 0.2, 0.3]
	assert processed_response.embeddings[0].index == 0
	assert processed_response.embeddings[1].values == [0.4, 0.5, 0.6]
	assert processed_response.embeddings[1].index == 1


def test_ollama_client_process_embeddings_response_empty():
	"""Test OllamaClient._process_embeddings_response with empty embeddings."""
	client = OllamaClient(
		ProviderModel(provider="ollama", api_key=None, base_url=None, model=None, timeout=60, max_retries=3)
	)
	request_model_name = "all-minilm"
	input_texts = []
	api_json_response = {
		"model": request_model_name,
		"embeddings": [],
	}
	validated_api_response = OllamaEmbeddingResponse.model_validate(api_json_response)
	processed_response = client._process_embeddings_response(validated_api_response, input_texts, request_model_name)
	assert isinstance(processed_response, TextEmbeddingResponse)
	assert len(processed_response.embeddings) == 0
	assert processed_response.texts == input_texts
	assert processed_response.model == request_model_name
	assert processed_response.provider == "ollama"
	assert processed_response.dimensions == 0


def test_ollama_client_embed_happy_path(httpx_mock):
	"""Test a basic successful call to OllamaClient.embed."""
	provider_config = ProviderModel(
		provider="ollama",
		api_key=None,
		base_url=None,
		model="all-minilm",
		timeout=60,
		max_retries=3,
	)
	ollama_client = OllamaClient(provider_config)
	input_text = "Test input for OllamaClient.embed"
	request_data = TextEmbeddingRequest(
		model=provider_config.model or ollama_client.default_model,
		input=input_text,
		dimensions=None,
		encoding_format=None,
		input_type=None,
		truncate=None,
	)
	mock_api_response_data = {
		"model": provider_config.model,
		"embeddings": [[0.5, 0.4, 0.3]],
		"total_duration": 1000,
		"load_duration": 100,
		"prompt_eval_count": 1,
	}
	httpx_mock.add_response(
		url="http://localhost:11434/api/embed",
		method="POST",
		json=mock_api_response_data,
		status_code=200,
	)
	with httpx.Client() as http_client:
		response = ollama_client.embed(request_data, http_client)
	assert isinstance(response, TextEmbeddingResponse)
	assert len(response.embeddings) == 1
	assert response.embeddings[0].values == [0.5, 0.4, 0.3]
	assert response.model == provider_config.model
	assert response.texts == [input_text]
	assert response.usage is not None
	assert response.usage["prompt_eval_count"] == 1


def test_ollama_client_embed_batch_happy_path(httpx_mock):
	"""Test a basic successful call to OllamaClient.embed_batch."""
	provider_config = ProviderModel(
		provider="ollama",
		api_key=None,
		base_url=None,
		model="all-minilm",
		timeout=60,
		max_retries=3,
	)
	ollama_client = OllamaClient(provider_config)
	input_texts = ["text1", "text2"]
	request_data = BatchEmbeddingRequest(
		model=provider_config.model or ollama_client.default_model,
		inputs=input_texts,
		batch_size=2,
		dimensions=None,
		encoding_format=None,
		input_type=None,
		truncate=None,
	)
	mock_api_response_data = {
		"model": provider_config.model,
		"embeddings": [[0.1, 0.2], [0.3, 0.4]],
		"total_duration": 2000,
		"load_duration": 200,
		"prompt_eval_count": 2,
	}
	httpx_mock.add_response(
		url="http://localhost:11434/api/embed",
		method="POST",
		json=mock_api_response_data,
		status_code=200,
	)
	with httpx.Client() as http_client:
		response = ollama_client.embed_batch(request_data, http_client)
	assert isinstance(response, BatchEmbeddingResponse)
	assert len(response.embeddings) == 2
	assert response.embeddings[0].values == [0.1, 0.2]
	assert response.embeddings[1].values == [0.3, 0.4]
	assert response.model == provider_config.model
	assert response.texts == input_texts
	assert response.failed_indices is None
	assert response.batch_count == 1


@pytest.mark.asyncio
async def test_ollama_client_aembed_happy_path(httpx_mock):
	"""Test a basic successful call to OllamaClient.aembed."""
	provider_config = ProviderModel(
		provider="ollama",
		api_key=None,
		base_url=None,
		model="all-minilm",
		timeout=60,
		max_retries=3,
	)
	ollama_client = OllamaClient(provider_config)
	input_text = "Test input for OllamaClient.aembed"
	request_data = TextEmbeddingRequest(
		model=provider_config.model or ollama_client.default_model,
		input=input_text,
		dimensions=None,
		encoding_format=None,
		input_type=None,
		truncate=None,
	)
	mock_api_response_data = {
		"model": provider_config.model,
		"embeddings": [[0.7, 0.8, 0.9]],
		"total_duration": 1500,
		"load_duration": 150,
		"prompt_eval_count": 1,
	}
	httpx_mock.add_response(
		url="http://localhost:11434/api/embed",
		method="POST",
		json=mock_api_response_data,
		status_code=200,
	)
	async with httpx.AsyncClient() as http_client:
		response = await ollama_client.aembed(request_data, http_client)
	assert isinstance(response, TextEmbeddingResponse)
	assert len(response.embeddings) == 1
	assert response.embeddings[0].values == [0.7, 0.8, 0.9]
	assert response.model == provider_config.model
	assert response.texts == [input_text]
	assert response.usage is not None
	assert response.usage["prompt_eval_count"] == 1


@pytest.mark.asyncio
async def test_ollama_client_aembed_batch_happy_path(httpx_mock):
	"""Test a basic successful call to OllamaClient.aembed_batch."""
	provider_config = ProviderModel(
		provider="ollama",
		api_key=None,
		base_url=None,
		model="all-minilm",
		timeout=60,
		max_retries=3,
	)
	ollama_client = OllamaClient(provider_config)
	input_texts = ["text1", "text2"]
	request_data = BatchEmbeddingRequest(
		model=provider_config.model or ollama_client.default_model,
		inputs=input_texts,
		batch_size=2,
		dimensions=None,
		encoding_format=None,
		input_type=None,
		truncate=None,
	)
	mock_api_response_data = {
		"model": provider_config.model,
		"embeddings": [[0.11, 0.12], [0.13, 0.14]],
		"total_duration": 2500,
		"load_duration": 250,
		"prompt_eval_count": 2,
	}
	httpx_mock.add_response(
		url="http://localhost:11434/api/embed",
		method="POST",
		json=mock_api_response_data,
		status_code=200,
	)
	async with httpx.AsyncClient() as http_client:
		response = await ollama_client.aembed_batch(request_data, http_client)
	assert isinstance(response, BatchEmbeddingResponse)
	assert len(response.embeddings) == 2
	assert response.embeddings[0].values == [0.11, 0.12]
	assert response.embeddings[1].values == [0.13, 0.14]
	assert response.model == provider_config.model
	assert response.texts == input_texts
	assert response.failed_indices is None
	assert response.batch_count == 1
