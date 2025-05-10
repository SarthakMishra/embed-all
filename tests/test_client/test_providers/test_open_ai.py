"""Tests for src/embed_all/client/providers/open_ai.py."""

import httpx
import pytest

from embed_all.client.providers.open_ai import OpenAIClient
from embed_all.models import (
	BatchEmbeddingRequest,
	BatchEmbeddingResponse,
	ProviderModel,
	TextEmbeddingRequest,
	TextEmbeddingResponse,
)
from embed_all.models.errors import AuthenticationError
from embed_all.models.responses.providers.open_ai import OpenAIAPIResponse


def test_openai_client_initialization_requires_api_key():
	"""Test OpenAIClient initialization raises AuthenticationError if api_key is not provided."""
	with pytest.raises(AuthenticationError, match=r"^\[openai\] API key is required for OpenAI$") as excinfo:
		OpenAIClient(
			ProviderModel(
				provider="openai",
				api_key=None,  # api_key is None
				base_url=None,
				model=None,
				timeout=60,
				max_retries=3,
			)
		)
	assert "[openai] API key is required for OpenAI" in str(excinfo.value)
	assert excinfo.value.provider == "openai"


def test_openai_client_initialization_with_api_key():
	"""Test OpenAIClient initialization succeeds with an API key."""
	try:
		OpenAIClient(
			ProviderModel(
				provider="openai",
				api_key="test_key",  # api_key is provided
				base_url=None,
				model=None,
				timeout=60,
				max_retries=3,
			)
		)
	except Exception as e:
		pytest.fail(f"OpenAIClient initialization failed unexpectedly: {e}")


def test_openai_client_get_headers():
	"""Test OpenAIClient._get_headers returns the correct headers."""
	api_key = "test_openai_api_key"
	client = OpenAIClient(
		ProviderModel(
			provider="openai",
			api_key=api_key,
			base_url=None,
			model=None,
			timeout=60,
			max_retries=3,
		)
	)
	headers = client._get_headers()  # Accessing private method for testing

	assert headers["Authorization"] == f"Bearer {api_key}"
	assert headers["Content-Type"] == "application/json"
	assert len(headers) == 2  # Ensure no other unexpected headers are present


def test_openai_client_process_embeddings_response_valid():
	"""Test OpenAIClient._process_embeddings_response with a valid API JSON response."""
	api_key = "test_key"
	client = OpenAIClient(
		ProviderModel(provider="openai", api_key=api_key, base_url=None, model=None, timeout=60, max_retries=3)
	)
	request_model_name = "text-embedding-ada-002"
	input_texts: list[str] = ["Hello world", "Another text"]

	api_json_response = {
		"object": "list",
		"data": [
			{
				"object": "embedding",
				"embedding": [0.1, 0.2, 0.3],
				"index": 0,
			},
			{
				"object": "embedding",
				"embedding": [0.4, 0.5, 0.6],
				"index": 1,
			},
		],
		"model": request_model_name,
		"usage": {"prompt_tokens": 5, "total_tokens": 5},
	}
	validated_api_response = OpenAIAPIResponse.model_validate(api_json_response)

	processed_response = client._process_embeddings_response(validated_api_response, input_texts)

	assert isinstance(processed_response, TextEmbeddingResponse)
	assert len(processed_response.embeddings) == 2
	assert processed_response.texts == input_texts
	assert processed_response.model == request_model_name
	assert processed_response.provider == "openai"
	assert processed_response.usage == {"prompt_tokens": 5, "total_tokens": 5}
	assert processed_response.embeddings[0].values == [0.1, 0.2, 0.3]
	assert processed_response.embeddings[0].index == 0
	assert processed_response.embeddings[1].values == [0.4, 0.5, 0.6]
	assert processed_response.embeddings[1].index == 1


def test_openai_client_process_embeddings_response_empty_data():
	"""Test OpenAIClient._process_embeddings_response with an API JSON response containing an empty 'data' list."""
	api_key = "test_key"
	client = OpenAIClient(
		ProviderModel(provider="openai", api_key=api_key, base_url=None, model=None, timeout=60, max_retries=3)
	)
	request_model_name = "text-embedding-3-large"
	input_texts: list[str] = []

	api_json_response = {
		"object": "list",
		"data": [],
		"model": request_model_name,
		"usage": {"prompt_tokens": 0, "total_tokens": 0},
	}
	validated_api_response = OpenAIAPIResponse.model_validate(api_json_response)

	# Expect a valid response even with empty data, not an APIError
	processed_response = client._process_embeddings_response(validated_api_response, input_texts)

	assert isinstance(processed_response, TextEmbeddingResponse)
	assert len(processed_response.embeddings) == 0
	assert processed_response.texts == input_texts
	assert processed_response.model == request_model_name
	assert processed_response.provider == "openai"
	assert processed_response.usage == {"prompt_tokens": 0, "total_tokens": 0}


# Tests for embed/aembed direct calls


@pytest.mark.asyncio
async def test_openai_client_aembed_happy_path(httpx_mock):
	"""Test a basic successful call to OpenAIClient.aembed."""
	api_key = "test_key_openai_aembed"
	provider_config = ProviderModel(
		provider="openai",
		api_key=api_key,
		base_url=None,
		model="text-embedding-ada-002",  # Default model for client
		timeout=60,
		max_retries=3,
	)
	openai_client = OpenAIClient(provider_config)

	input_text = "Test input for OpenAIClient.aembed"
	# Ensure a string model is passed to TextEmbeddingRequest
	effective_model_aembed = provider_config.model or openai_client.default_model
	request_data = TextEmbeddingRequest(
		model=effective_model_aembed,
		input=input_text,
		dimensions=None,
		encoding_format=None,
		input_type=None,
		truncate=None,
	)

	mock_api_response_data = {
		"object": "list",
		"data": [
			{
				"object": "embedding",
				"embedding": [0.5, 0.4, 0.3],
				"index": 0,
			}
		],
		"model": provider_config.model,
		"usage": {"prompt_tokens": 7, "total_tokens": 7},
	}
	httpx_mock.add_response(
		url="https://api.openai.com/v1/embeddings",  # Default URL OpenAIClient uses
		method="POST",
		json=mock_api_response_data,
		status_code=200,
	)

	async with httpx.AsyncClient() as http_client:  # The client OpenAIClient.aembed expects
		response = await openai_client.aembed(request_data, http_client)

	assert isinstance(response, TextEmbeddingResponse)
	assert len(response.embeddings) == 1
	assert response.embeddings[0].values == [0.5, 0.4, 0.3]
	assert response.model == provider_config.model
	assert response.texts == [input_text]
	assert response.usage == {"prompt_tokens": 7, "total_tokens": 7}


@pytest.mark.asyncio
async def test_openai_client_embed_happy_path(httpx_mock):
	"""Test a basic successful call to OpenAIClient.embed."""
	api_key = "test_key_openai_embed"
	provider_config = ProviderModel(
		provider="openai",
		api_key=api_key,
		base_url=None,
		model="text-embedding-3-small",  # Default model for client
		timeout=60,
		max_retries=3,
	)
	openai_client = OpenAIClient(provider_config)

	input_text = "Test input for OpenAIClient.embed"
	# Ensure a string model is passed to TextEmbeddingRequest
	effective_model_embed = provider_config.model or openai_client.default_model
	request_data = TextEmbeddingRequest(
		model=effective_model_embed,
		input=input_text,
		dimensions=None,
		encoding_format=None,
		input_type=None,
		truncate=None,
	)

	mock_api_response_data = {
		"object": "list",
		"data": [
			{
				"object": "embedding",
				"embedding": [0.6, 0.7, 0.8],
				"index": 0,
			}
		],
		"model": provider_config.model,
		"usage": {"prompt_tokens": 8, "total_tokens": 8},
	}
	httpx_mock.add_response(
		url="https://api.openai.com/v1/embeddings",
		method="POST",
		json=mock_api_response_data,
		status_code=200,
	)

	with httpx.Client() as http_client:  # The client OpenAIClient.embed expects
		response = openai_client.embed(request_data, http_client)

	assert isinstance(response, TextEmbeddingResponse)
	assert len(response.embeddings) == 1
	assert response.embeddings[0].values == [0.6, 0.7, 0.8]
	assert response.model == provider_config.model
	assert response.texts == [input_text]
	assert response.usage == {"prompt_tokens": 8, "total_tokens": 8}


# Tests for embed_batch/aembed_batch direct calls (OpenAI-specific batching)


@pytest.mark.asyncio
async def test_openai_client_aembed_batch_less_than_max_size(httpx_mock):
	"""Test OpenAIClient.aembed_batch with input size < OPENAI_API_MAX_BATCH_SIZE."""
	api_key = "test_key_openai_aembed_batch_small"
	model_name = "text-embedding-ada-002"
	provider_config = ProviderModel(
		provider="openai", api_key=api_key, base_url=None, model=model_name, timeout=60, max_retries=3
	)
	openai_client = OpenAIClient(provider_config)

	# OPENAI_API_MAX_BATCH_SIZE is 2048. Let's use a small number like 2.
	input_texts = ["text 1", "text 2"]
	request_data = BatchEmbeddingRequest(
		model=model_name,
		inputs=input_texts,
		dimensions=None,  # Add all optional fields for BatchEmbeddingRequest
		batch_size=32,  # This batch_size is for the client, not the API limit
		encoding_format=None,
		input_type=None,
		truncate=None,
	)

	mock_api_response_data = {
		"object": "list",
		"data": [
			{"object": "embedding", "embedding": [0.1] * 10, "index": 0},
			{"object": "embedding", "embedding": [0.2] * 10, "index": 1},
		],
		"model": model_name,
		"usage": {"prompt_tokens": 4, "total_tokens": 4},
	}
	# Expect one API call as len(input_texts) < OPENAI_API_MAX_BATCH_SIZE
	httpx_mock.add_response(
		url="https://api.openai.com/v1/embeddings",
		method="POST",
		json=mock_api_response_data,
		status_code=200,
		# We can also assert the request body if needed, e.g., via match_content
		# match_content=b'{"input": ["text 1", "text 2"], "model": "text-embedding-ada-002"}'
	)

	async with httpx.AsyncClient() as http_client:
		response = await openai_client.aembed_batch(request_data, http_client)

	assert isinstance(response, BatchEmbeddingResponse)
	assert len(response.embeddings) == 2
	assert response.texts == input_texts
	assert response.model == model_name
	assert response.usage == {"prompt_tokens": 4, "total_tokens": 4}
	assert response.failed_indices is None or len(response.failed_indices) == 0
	assert httpx_mock.get_requests() is not None
	assert len(httpx_mock.get_requests()) == 1  # Verify only one API call was made


@pytest.mark.asyncio
async def test_openai_client_aembed_batch_more_than_max_size(httpx_mock, mocker):
	"""Test OpenAIClient.aembed_batch with input size > OPENAI_API_MAX_BATCH_SIZE, forcing multiple API calls."""
	# Patch the constant OPENAI_API_MAX_BATCH_SIZE within the provider module for this test
	mocker.patch("embed_all.client.providers.open_ai.OPENAI_API_MAX_BATCH_SIZE", 1)

	api_key = "test_key_openai_aembed_batch_large"
	model_name = "text-embedding-ada-002"
	provider_config = ProviderModel(
		provider="openai", api_key=api_key, base_url=None, model=model_name, timeout=60, max_retries=3
	)
	openai_client = OpenAIClient(provider_config)

	input_texts = ["text one", "text two", "text three"]  # Will require 3 API calls
	request_data = BatchEmbeddingRequest(
		model=model_name,
		inputs=input_texts,
		dimensions=None,
		batch_size=32,  # Client-side batch size, not relevant for this API limit test
		encoding_format=None,
		input_type=None,
		truncate=None,
	)

	# Mock responses for each API call
	# Call 1 (for "text one")
	httpx_mock.add_response(
		url="https://api.openai.com/v1/embeddings",
		method="POST",
		status_code=200,
		json={
			"object": "list",
			"data": [{"object": "embedding", "embedding": [0.1] * 5, "index": 0}],
			"model": model_name,
			"usage": {"prompt_tokens": 1, "total_tokens": 1},
		},
	)
	# Call 2 (for "text two")
	httpx_mock.add_response(
		url="https://api.openai.com/v1/embeddings",
		method="POST",
		status_code=200,
		json={
			"object": "list",
			"data": [{"object": "embedding", "embedding": [0.2] * 5, "index": 0}],
			"model": model_name,
			"usage": {"prompt_tokens": 1, "total_tokens": 1},
		},
	)
	# Call 3 (for "text three")
	httpx_mock.add_response(
		url="https://api.openai.com/v1/embeddings",
		method="POST",
		status_code=200,
		json={
			"object": "list",
			"data": [{"object": "embedding", "embedding": [0.3] * 5, "index": 0}],
			"model": model_name,
			"usage": {"prompt_tokens": 1, "total_tokens": 1},
		},
	)

	async with httpx.AsyncClient() as http_client:
		response = await openai_client.aembed_batch(request_data, http_client)

	assert isinstance(response, BatchEmbeddingResponse)
	assert len(response.embeddings) == 3
	assert response.texts == input_texts
	assert response.model == model_name
	assert response.usage == {"prompt_tokens": 3, "total_tokens": 3}  # Sum of usages from 3 calls
	assert response.failed_indices is None or len(response.failed_indices) == 0

	assert response.embeddings[0].values == [0.1] * 5
	assert response.embeddings[0].index == 0  # Original index
	assert response.embeddings[1].values == [0.2] * 5
	assert response.embeddings[1].index == 1  # Original index
	assert response.embeddings[2].values == [0.3] * 5
	assert response.embeddings[2].index == 2  # Original index

	assert httpx_mock.get_requests() is not None
	assert len(httpx_mock.get_requests()) == 3  # Verify three API calls were made


@pytest.mark.asyncio
async def test_openai_client_aembed_batch_partial_failure_multi_calls(httpx_mock, mocker):
	"""Test OpenAIClient.aembed_batch with one internal call failing and others succeeding."""
	mocker.patch("embed_all.client.providers.open_ai.OPENAI_API_MAX_BATCH_SIZE", 1)  # Each text is a batch

	api_key = "test_key_openai_aembed_batch_partial_fail"
	model_name = "text-embedding-ada-002"
	provider_config = ProviderModel(
		provider="openai", api_key=api_key, base_url=None, model=model_name, timeout=60, max_retries=3
	)
	openai_client = OpenAIClient(provider_config)

	input_texts = ["text success 1", "text fail 2", "text success 3"]
	request_data = BatchEmbeddingRequest(
		model=model_name,
		inputs=input_texts,
		dimensions=None,
		batch_size=32,
		encoding_format=None,
		input_type=None,
		truncate=None,
	)

	# Mock responses
	# Call 1 (success for "text success 1")
	httpx_mock.add_response(
		url="https://api.openai.com/v1/embeddings",
		method="POST",
		status_code=200,
		json={
			"object": "list",
			"data": [{"object": "embedding", "embedding": [0.1] * 5, "index": 0}],
			"model": model_name,
			"usage": {"prompt_tokens": 1, "total_tokens": 1},
		},
	)
	# Call 2 (failure for "text fail 2")
	httpx_mock.add_response(
		url="https://api.openai.com/v1/embeddings",
		method="POST",
		status_code=400,  # Bad request
		json={"error": {"message": "Invalid input for text fail 2"}},
	)
	# Call 3 (success for "text success 3")
	httpx_mock.add_response(
		url="https://api.openai.com/v1/embeddings",
		method="POST",
		status_code=200,
		json={
			"object": "list",
			"data": [{"object": "embedding", "embedding": [0.3] * 5, "index": 0}],
			"model": model_name,
			"usage": {"prompt_tokens": 1, "total_tokens": 1},
		},
	)

	# Ensure OpenAIAPIResponse.model_validate behaves normally for successful calls
	# The failing call (batch 2) won't reach model_validate due to raise_for_status() in the provider.
	original_model_validate = OpenAIAPIResponse.model_validate
	mocker.patch(
		"embed_all.models.responses.providers.open_ai.OpenAIAPIResponse.model_validate",
		side_effect=lambda data, **kwargs: original_model_validate(data, **kwargs),
	)

	async with httpx.AsyncClient() as http_client:
		response = await openai_client.aembed_batch(request_data, http_client)

	assert isinstance(response, BatchEmbeddingResponse)
	assert len(response.embeddings) == 2  # Two successes
	assert response.failed_indices is not None
	assert len(response.failed_indices) == 1
	assert 1 in response.failed_indices

	assert response.model == model_name
	assert response.texts == input_texts  # Should contain all original texts
	assert response.usage == {"prompt_tokens": 2, "total_tokens": 2}  # Sum of 2 successful calls

	# Check successful embeddings (original indices 0 and 2)
	assert response.embeddings[0].values == [0.1] * 5
	assert response.embeddings[0].index == 0
	assert response.embeddings[1].values == [0.3] * 5
	assert response.embeddings[1].index == 2

	assert httpx_mock.get_requests() is not None
	assert len(httpx_mock.get_requests()) == 3  # All three API calls attempted


# Add more tests for OpenAIClient methods below
