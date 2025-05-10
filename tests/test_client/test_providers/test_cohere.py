"""Tests for src/embed_all/client/providers/cohere.py."""

import httpx
import pytest

from embed_all.client.providers.cohere import CohereClient
from embed_all.models import (
	BatchEmbeddingRequest,
	BatchEmbeddingResponse,
	ProviderModel,
	TextEmbeddingRequest,
	TextEmbeddingResponse,
)
from embed_all.models.responses.providers.cohere import CohereAPIResponse


def test_cohere_client_initialization():
	"""Test CohereClient initialization with and without API key."""
	# API key is optional for Cohere, but test both cases
	client = CohereClient(
		ProviderModel(
			provider="cohere",
			api_key="test_key",
			base_url=None,
			model=None,
			timeout=60,
			max_retries=3,
		)
	)
	assert client.base_url == "https://api.cohere.com"
	assert client.default_model == "embed-v4.0"


def test_cohere_client_get_headers():
	"""Test CohereClient._get_headers returns the correct headers."""
	api_key = "test_cohere_api_key"
	client = CohereClient(
		ProviderModel(
			provider="cohere",
			api_key=api_key,
			base_url=None,
			model=None,
			timeout=60,
			max_retries=3,
		)
	)
	headers = client._get_headers()
	assert headers["Authorization"] == f"bearer {api_key}"
	assert headers["Content-Type"] == "application/json"
	assert headers["Accept"] == "application/json"
	assert len(headers) == 3


def test_cohere_client_process_embeddings_response_valid():
	"""Test CohereClient._process_embeddings_response with a valid API JSON response."""
	api_key = "test_key"
	client = CohereClient(
		ProviderModel(provider="cohere", api_key=api_key, base_url=None, model=None, timeout=60, max_retries=3)
	)
	request_model_name = "embed-v4.0"
	input_texts = ["Hello world", "Another text"]
	api_json_response = {
		"id": "abc123",
		"texts": input_texts,
		"embeddings": {"float": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]},
		"meta": {
			"api_version": {"version": "2", "is_experimental": True},
			"billed_units": {"input_tokens": 5},
		},
	}
	validated_api_response = CohereAPIResponse.model_validate(api_json_response)
	processed_response = client._process_embeddings_response(validated_api_response, input_texts, request_model_name)
	assert isinstance(processed_response, TextEmbeddingResponse)
	assert len(processed_response.embeddings) == 2
	assert processed_response.texts == input_texts
	assert processed_response.model == request_model_name
	assert processed_response.provider == "cohere"
	assert processed_response.usage == {"tokens": 5}
	assert processed_response.embeddings[0].values == [0.1, 0.2, 0.3]
	assert processed_response.embeddings[0].index == 0
	assert processed_response.embeddings[1].values == [0.4, 0.5, 0.6]
	assert processed_response.embeddings[1].index == 1


def test_cohere_client_process_embeddings_response_empty_data():
	"""Test CohereClient._process_embeddings_response with an API JSON response containing an empty 'float' list."""
	api_key = "test_key"
	client = CohereClient(
		ProviderModel(provider="cohere", api_key=api_key, base_url=None, model=None, timeout=60, max_retries=3)
	)
	request_model_name = "embed-v4.0"
	input_texts = []
	api_json_response = {
		"id": "abc123",
		"texts": [],
		"embeddings": {"float": []},
		"meta": {"api_version": {"version": "2", "is_experimental": True}, "billed_units": {"input_tokens": 0}},
	}
	validated_api_response = CohereAPIResponse.model_validate(api_json_response)
	processed_response = client._process_embeddings_response(validated_api_response, input_texts, request_model_name)
	assert isinstance(processed_response, TextEmbeddingResponse)
	assert len(processed_response.embeddings) == 0
	assert processed_response.texts == input_texts
	assert processed_response.model == request_model_name
	assert processed_response.provider == "cohere"
	assert processed_response.usage == {"tokens": 0}


@pytest.mark.asyncio
async def test_cohere_client_aembed_happy_path(httpx_mock):
	"""Test a basic successful call to CohereClient.aembed."""
	api_key = "test_key_cohere_aembed"
	provider_config = ProviderModel(
		provider="cohere",
		api_key=api_key,
		base_url=None,
		model="embed-v4.0",
		timeout=60,
		max_retries=3,
	)
	cohere_client = CohereClient(provider_config)
	input_text = "Test input for CohereClient.aembed"
	request_data = TextEmbeddingRequest(
		model=provider_config.model or cohere_client.default_model,
		input=input_text,
		dimensions=None,
		encoding_format=None,
		input_type=None,
		truncate=None,
	)
	mock_api_response_data = {
		"id": "abc456",
		"texts": [input_text],
		"embeddings": {"float": [[0.5, 0.4, 0.3]]},
		"meta": {"api_version": {"version": "2"}, "billed_units": {"input_tokens": 7}},
	}
	httpx_mock.add_response(
		url="https://api.cohere.com/v2/embed",
		method="POST",
		json=mock_api_response_data,
		status_code=200,
	)
	async with httpx.AsyncClient() as http_client:
		response = await cohere_client.aembed(request_data, http_client)
	assert isinstance(response, TextEmbeddingResponse)
	assert len(response.embeddings) == 1
	assert response.embeddings[0].values == [0.5, 0.4, 0.3]
	assert response.model == provider_config.model
	assert response.texts == [input_text]
	assert response.usage == {"tokens": 7}


def test_cohere_client_embed_happy_path(httpx_mock):
	"""Test a basic successful call to CohereClient.embed."""
	api_key = "test_key_cohere_embed"
	provider_config = ProviderModel(
		provider="cohere",
		api_key=api_key,
		base_url=None,
		model="embed-v4.0",
		timeout=60,
		max_retries=3,
	)
	cohere_client = CohereClient(provider_config)
	input_text = "Test input for CohereClient.embed"
	request_data = TextEmbeddingRequest(
		model=provider_config.model or cohere_client.default_model,
		input=input_text,
		dimensions=None,
		encoding_format=None,
		input_type=None,
		truncate=None,
	)
	mock_api_response_data = {
		"id": "abc789",
		"texts": [input_text],
		"embeddings": {"float": [[0.6, 0.7, 0.8]]},
		"meta": {"api_version": {"version": "2"}, "billed_units": {"input_tokens": 8}},
	}
	httpx_mock.add_response(
		url="https://api.cohere.com/v2/embed",
		method="POST",
		json=mock_api_response_data,
		status_code=200,
	)
	with httpx.Client() as http_client:
		response = cohere_client.embed(request_data, http_client)
	assert isinstance(response, TextEmbeddingResponse)
	assert len(response.embeddings) == 1
	assert response.embeddings[0].values == [0.6, 0.7, 0.8]
	assert response.model == provider_config.model
	assert response.texts == [input_text]
	assert response.usage == {"tokens": 8}


@pytest.mark.asyncio
async def test_cohere_client_aembed_batch_less_than_max_size(httpx_mock):
	"""Test CohereClient.aembed_batch with input size < COHERE_API_MAX_BATCH_SIZE."""
	api_key = "test_key_cohere_aembed_batch_small"
	model_name = "embed-v4.0"
	provider_config = ProviderModel(
		provider="cohere", api_key=api_key, base_url=None, model=model_name, timeout=60, max_retries=3
	)
	cohere_client = CohereClient(provider_config)
	input_texts = ["text 1", "text 2"]
	request_data = BatchEmbeddingRequest(
		model=model_name,
		inputs=input_texts,
		dimensions=None,
		batch_size=32,
		encoding_format=None,
		input_type=None,
		truncate=None,
	)
	mock_api_response_data = {
		"id": "batch123",
		"texts": input_texts,
		"embeddings": {"float": [[0.1] * 10, [0.2] * 10]},
		"meta": {"api_version": {"version": "2"}, "billed_units": {"input_tokens": 4}},
	}
	httpx_mock.add_response(
		url="https://api.cohere.com/v2/embed",
		method="POST",
		json=mock_api_response_data,
		status_code=200,
	)
	async with httpx.AsyncClient() as http_client:
		response = await cohere_client.aembed_batch(request_data, http_client)
	assert isinstance(response, BatchEmbeddingResponse)
	assert len(response.embeddings) == 2
	assert response.texts == input_texts
	assert response.model == model_name
	assert response.usage == {"tokens": 4}
	assert response.failed_indices is None or len(response.failed_indices) == 0
	assert httpx_mock.get_requests() is not None
	assert len(httpx_mock.get_requests()) == 1


@pytest.mark.asyncio
async def test_cohere_client_aembed_batch_partial_failure(httpx_mock):
	"""Test CohereClient.aembed_batch with one batch failing and others succeeding."""
	api_key = "test_key_cohere_aembed_batch_partial_fail"
	model_name = "embed-v4.0"
	provider_config = ProviderModel(
		provider="cohere", api_key=api_key, base_url=None, model=model_name, timeout=60, max_retries=3
	)
	cohere_client = CohereClient(provider_config)
	input_texts = ["text success 1", "text fail 2", "text success 3"]
	request_data = BatchEmbeddingRequest(
		model=model_name,
		inputs=input_texts,
		dimensions=None,
		batch_size=1,  # Force each text to be a separate batch
		encoding_format=None,
		input_type=None,
		truncate=None,
	)
	# Batch 1: success
	httpx_mock.add_response(
		url="https://api.cohere.com/v2/embed",
		method="POST",
		status_code=200,
		json={
			"id": "batch1",
			"texts": ["text success 1"],
			"embeddings": {"float": [[0.1] * 5]},
			"meta": {"api_version": {"version": "2"}, "billed_units": {"input_tokens": 1}},
		},
	)
	# Batch 2: failure
	httpx_mock.add_response(
		url="https://api.cohere.com/v2/embed",
		method="POST",
		status_code=400,
		json={"error": {"message": "Invalid input for text fail 2"}},
	)
	# Batch 3: success
	httpx_mock.add_response(
		url="https://api.cohere.com/v2/embed",
		method="POST",
		status_code=200,
		json={
			"id": "batch3",
			"texts": ["text success 3"],
			"embeddings": {"float": [[0.3] * 5]},
			"meta": {"api_version": {"version": "2"}, "billed_units": {"input_tokens": 1}},
		},
	)
	async with httpx.AsyncClient() as http_client:
		response = await cohere_client.aembed_batch(request_data, http_client)
	assert isinstance(response, BatchEmbeddingResponse)
	assert len(response.embeddings) == 2  # Two successes
	assert response.failed_indices is not None
	assert len(response.failed_indices) == 1
	assert 1 in response.failed_indices
	assert response.model == model_name
	assert response.texts == input_texts
	assert response.usage == {"tokens": 2}
	assert httpx_mock.get_requests() is not None
	assert len(httpx_mock.get_requests()) == 3


def test_cohere_client_embed_batch_less_than_max_size(httpx_mock):
	"""Test CohereClient.embed_batch with input size < COHERE_API_MAX_BATCH_SIZE."""
	api_key = "test_key_cohere_embed_batch_small"
	model_name = "embed-v4.0"
	provider_config = ProviderModel(
		provider="cohere", api_key=api_key, base_url=None, model=model_name, timeout=60, max_retries=3
	)
	cohere_client = CohereClient(provider_config)
	input_texts = ["sync batch text 1", "sync batch text 2"]
	request_data = BatchEmbeddingRequest(
		model=model_name,
		inputs=input_texts,
		dimensions=None,
		batch_size=32,
		encoding_format=None,
		input_type=None,
		truncate=None,
	)
	mock_api_response_data = {
		"id": "batchsync123",
		"texts": input_texts,
		"embeddings": {"float": [[0.5] * 10, [0.6] * 10]},
		"meta": {"api_version": {"version": "2"}, "billed_units": {"input_tokens": 5}},
	}
	httpx_mock.add_response(
		url="https://api.cohere.com/v2/embed",
		method="POST",
		json=mock_api_response_data,
		status_code=200,
	)
	with httpx.Client() as http_client:
		response = cohere_client.embed_batch(request_data, http_client)
	assert isinstance(response, BatchEmbeddingResponse)
	assert len(response.embeddings) == 2
	assert response.texts == input_texts
	assert response.model == model_name
	assert response.usage == {"tokens": 5}
	assert not response.failed_indices
	assert len(httpx_mock.get_requests()) == 1


def test_cohere_client_embed_batch_partial_failure(httpx_mock):
	"""Test CohereClient.embed_batch with partial failure."""
	api_key = "test_key_cohere_embed_batch_partial_fail"
	model_name = "embed-v4.0"
	provider_config = ProviderModel(
		provider="cohere", api_key=api_key, model=model_name, base_url=None, timeout=60, max_retries=3
	)
	cohere_client = CohereClient(provider_config)
	input_texts = ["sync success 1", "sync fail 2", "sync success 3"]
	request_data = BatchEmbeddingRequest(
		model=model_name,
		inputs=input_texts,
		dimensions=None,
		batch_size=1,
		encoding_format=None,
		input_type=None,
		truncate=None,
	)
	# Batch 1: success
	httpx_mock.add_response(
		url="https://api.cohere.com/v2/embed",
		method="POST",
		status_code=200,
		json={
			"id": "batch1",
			"texts": ["sync success 1"],
			"embeddings": {"float": [[0.1] * 5]},
			"meta": {"api_version": {"version": "2"}, "billed_units": {"input_tokens": 1}},
		},
	)
	# Batch 2: failure
	httpx_mock.add_response(
		url="https://api.cohere.com/v2/embed",
		method="POST",
		status_code=400,
		json={"error": {"message": "Invalid for sync fail 2"}},
	)
	# Batch 3: success
	httpx_mock.add_response(
		url="https://api.cohere.com/v2/embed",
		method="POST",
		status_code=200,
		json={
			"id": "batch3",
			"texts": ["sync success 3"],
			"embeddings": {"float": [[0.3] * 5]},
			"meta": {"api_version": {"version": "2"}, "billed_units": {"input_tokens": 1}},
		},
	)
	with httpx.Client() as http_client:
		response = cohere_client.embed_batch(request_data, http_client)
	assert isinstance(response, BatchEmbeddingResponse)
	assert len(response.embeddings) == 2
	assert response.failed_indices is not None
	failed_list_for_check = response.failed_indices
	assert len(failed_list_for_check) == 1
	assert 1 in failed_list_for_check
	assert response.model == model_name
	assert response.texts == input_texts
	assert response.usage == {"tokens": 2}
	assert len(httpx_mock.get_requests()) == 3
