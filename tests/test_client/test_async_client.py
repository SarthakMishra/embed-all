"""Tests for embed_all.client.async_client module."""

import logging
from typing import Any, cast

import httpx
import pytest

from embed_all.client.async_client import AsyncClient
from embed_all.client.providers.open_ai import OpenAIClient
from embed_all.models import BatchEmbeddingRequest
from embed_all.models.base import EmbeddingModel
from embed_all.models.errors import APIError, AuthenticationError, EmbedError, InvalidRequestError, RateLimitError
from embed_all.models.responses.batch import BatchEmbeddingResponse
from embed_all.models.responses.providers.open_ai import OpenAIAPIResponse
from embed_all.models.responses.text import TextEmbeddingResponse


def test_async_client_initialization():
	"""Test that AsyncClient initializes correctly."""
	client = AsyncClient(provider="openai", api_key="test_key")
	assert client.config.provider == "openai"
	assert client.config.api_key == "test_key"
	assert client.config.model == "text-embedding-3-small"


@pytest.mark.asyncio
async def test_aembed_simple_text(httpx_mock):
	"""Test aembed with a single text string."""
	provider_name = "openai"
	api_key = "fake_api_key"
	request_model = "text-embedding-ada-002"

	mock_response_data = {
		"object": "list",
		"data": [
			{
				"object": "embedding",
				"embedding": [0.1, 0.2, 0.3],
				"index": 0,
			}
		],
		"model": request_model,
		"usage": {"prompt_tokens": 5, "total_tokens": 5},
	}
	httpx_mock.add_response(
		url="https://api.openai.com/v1/embeddings",
		method="POST",
		json=mock_response_data,
		status_code=200,
	)

	async with AsyncClient(provider=provider_name, api_key=api_key, model=request_model) as client:
		response = await client.aembed(text="Hello, world!")

	assert isinstance(response, TextEmbeddingResponse)
	assert len(response.embeddings) == 1
	assert isinstance(response.embeddings[0], EmbeddingModel)
	assert response.embeddings[0].values == [0.1, 0.2, 0.3]
	assert response.model == request_model
	assert response.provider == provider_name
	assert response.usage is not None
	assert response.usage["prompt_tokens"] == 5
	assert response.usage["total_tokens"] == 5


@pytest.mark.asyncio
async def test_aembed_authentication_error(httpx_mock):
	"""Test aembed raises AuthenticationError on 401."""
	provider_name = "openai"
	api_key = "invalid_key"
	request_model = "text-embedding-ada-002"

	httpx_mock.add_response(
		url="https://api.openai.com/v1/embeddings",
		method="POST",
		status_code=401,
		json={"error": {"message": "Incorrect API key"}},
	)

	with pytest.raises(AuthenticationError) as excinfo:
		async with AsyncClient(provider=provider_name, api_key=api_key, model=request_model) as client:
			await client.aembed(text="Hello, world!")

	assert "[openai] Authentication failed (Status: 401)" in str(excinfo.value)
	assert excinfo.value.provider == provider_name
	assert excinfo.value.status_code == 401


@pytest.mark.asyncio
async def test_aembed_rate_limit_error(httpx_mock):
	"""Test aembed raises RateLimitError on 429."""
	provider_name = "openai"
	api_key = "fake_key"
	request_model = "text-embedding-ada-002"

	httpx_mock.add_response(
		url="https://api.openai.com/v1/embeddings",
		method="POST",
		status_code=429,
		json={"error": {"message": "Rate limit exceeded"}},
		headers={"retry-after": "60"},
	)

	with pytest.raises(RateLimitError) as excinfo:
		async with AsyncClient(provider=provider_name, api_key=api_key, model=request_model) as client:
			await client.aembed(text="Hello, world!")

	assert "[openai] Rate limit exceeded (Status: 429)" in str(excinfo.value)
	assert excinfo.value.provider == provider_name
	assert excinfo.value.status_code == 429
	assert excinfo.value.retry_after == 60


@pytest.mark.asyncio
async def test_aembed_invalid_request_error(httpx_mock):
	"""Test aembed raises InvalidRequestError on 400."""
	provider_name = "openai"
	api_key = "fake_key"
	request_model = "text-embedding-ada-002"

	httpx_mock.add_response(
		url="https://api.openai.com/v1/embeddings",
		method="POST",
		status_code=400,
		json={"error": {"message": "Invalid request data"}},
	)

	with pytest.raises(InvalidRequestError) as excinfo:
		async with AsyncClient(provider=provider_name, api_key=api_key, model=request_model) as client:
			await client.aembed(text="Hello, world!")

	assert "[openai] Invalid request data (Status: 400)" in str(excinfo.value)
	assert excinfo.value.provider == provider_name
	assert excinfo.value.status_code == 400


@pytest.mark.asyncio
async def test_aembed_generic_api_error(httpx_mock):
	"""Test aembed raises APIError on other HTTP errors (e.g., 500)."""
	provider_name = "openai"
	api_key = "fake_key"
	request_model = "text-embedding-ada-002"

	httpx_mock.add_response(
		url="https://api.openai.com/v1/embeddings",
		method="POST",
		status_code=500,
		json={"error": {"message": "Internal server error"}},
	)

	with pytest.raises(APIError) as excinfo:
		async with AsyncClient(provider=provider_name, api_key=api_key, model=request_model) as client:
			await client.aembed(text="Hello, world!")

	assert "[openai] Internal server error (Status: 500)" in str(excinfo.value)
	assert excinfo.value.provider == provider_name
	assert excinfo.value.status_code == 500
	# Check that it's not a more specific error type
	assert not isinstance(excinfo.value, AuthenticationError)
	assert not isinstance(excinfo.value, RateLimitError)
	assert not isinstance(excinfo.value, InvalidRequestError)


@pytest.mark.asyncio
async def test_aembed_non_embed_error_from_provider(mocker):
	"""Test aembed wraps non-EmbedError from provider into EmbedError."""
	provider_name = "openai"
	api_key = "fake_key"
	request_model = "text-embedding-3-small"

	# Mock the provider client's aembed to raise a generic Exception
	mock_provider_client = mocker.MagicMock()
	# We need to ensure the mock's aembed is an async function if it's awaited
	# For side_effect raising an exception, it doesn't strictly need to be async,
	# but if we were returning a value, it would.
	mock_provider_client.aembed.side_effect = ValueError("Provider internal error")

	# We need to mock get_provider_client to return our mock_provider_client
	# when AsyncClient initializes it.
	mocker.patch("embed_all.client.async_client.get_provider_client", return_value=lambda _: mock_provider_client)

	async with AsyncClient(provider=provider_name, api_key=api_key, model=request_model) as client:
		with pytest.raises(EmbedError) as excinfo:
			await client.aembed(text="Hello, error world!")

	assert "Error generating embeddings: Provider internal error" in str(excinfo.value)
	assert excinfo.value.provider == provider_name
	assert not isinstance(excinfo.value, (AuthenticationError, RateLimitError, InvalidRequestError))
	# Ensure the original error is chained
	assert isinstance(excinfo.value.__cause__, ValueError)
	assert str(excinfo.value.__cause__) == "Provider internal error"


@pytest.mark.asyncio
async def test_aembed_batch_simple(httpx_mock):
	"""Test aembed_batch with a simple batch of texts."""
	provider_name = "openai"
	api_key = "fake_api_key"
	request_model = "text-embedding-ada-002"
	texts_to_embed = ["Hello, first!", "Hello, second!"]

	# Mocking a response for a batch request.
	# OpenAI typically returns embeddings in the same order as input.
	mock_response_data = {
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
		"model": request_model,
		"usage": {"prompt_tokens": 10, "total_tokens": 10},  # Example usage for two texts
	}
	httpx_mock.add_response(
		url="https://api.openai.com/v1/embeddings",
		method="POST",
		json=mock_response_data,
		status_code=200,
	)

	async with AsyncClient(provider=provider_name, api_key=api_key, model=request_model) as client:
		response = await client.aembed_batch(texts=texts_to_embed, batch_size=2)

	assert isinstance(response, BatchEmbeddingResponse)
	assert len(response.embeddings) == 2
	assert len(response.texts) == 2  # Should match the input texts count
	assert isinstance(response.embeddings[0], EmbeddingModel)
	assert response.embeddings[0].values == [0.1, 0.2, 0.3]
	# assert response.embeddings[0].text == texts_to_embed[0] # This depends on provider client populating it
	assert isinstance(response.embeddings[1], EmbeddingModel)
	assert response.embeddings[1].values == [0.4, 0.5, 0.6]
	# assert response.embeddings[1].text == texts_to_embed[1] # This depends on provider client populating it

	assert response.model == request_model
	assert response.provider == provider_name
	assert response.failed_indices is None or len(response.failed_indices) == 0
	assert response.usage is not None
	assert response.usage["prompt_tokens"] == 10
	assert response.usage["total_tokens"] == 10


@pytest.mark.asyncio
async def test_aembed_list_input_happy_path(httpx_mock):
	"""Test aembed with a list of text strings - happy path."""
	provider_name = "openai"
	api_key = "fake_api_key"
	request_model = "text-embedding-ada-002"
	texts_to_embed = ["Hello, async list world!", "Another text for async list."]

	mock_response_data = {
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
		"model": request_model,
		"usage": {"prompt_tokens": 10, "total_tokens": 10},  # Example usage
	}
	httpx_mock.add_response(
		url="https://api.openai.com/v1/embeddings",
		method="POST",
		json=mock_response_data,
		status_code=200,
	)

	async with AsyncClient(provider=provider_name, api_key=api_key, model=request_model) as client:
		response = await client.aembed(text=texts_to_embed)

	assert isinstance(response, TextEmbeddingResponse)
	assert len(response.embeddings) == len(texts_to_embed)
	assert response.texts == texts_to_embed
	assert response.model == request_model
	assert response.provider == provider_name
	assert response.usage is not None
	assert response.usage["prompt_tokens"] == 10
	assert response.usage["total_tokens"] == 10

	for i, embedding in enumerate(response.embeddings):
		assert isinstance(embedding, EmbeddingModel)
		assert embedding.index == i
		if i == 0:
			assert embedding.values == [0.1, 0.2, 0.3]
		elif i == 1:
			assert embedding.values == [0.4, 0.5, 0.6]


@pytest.mark.asyncio
async def test_aembed_list_input_authentication_error(httpx_mock):
	"""Test aembed with a list of text strings raises AuthenticationError."""
	provider_name = "openai"
	api_key = "invalid_key"
	request_model = "text-embedding-ada-002"
	texts_to_embed = ["text1", "text2"]

	httpx_mock.add_response(
		url="https://api.openai.com/v1/embeddings",
		method="POST",
		status_code=401,
		json={"error": {"message": "Incorrect API key provided"}},
	)

	with pytest.raises(AuthenticationError) as excinfo:
		async with AsyncClient(provider=provider_name, api_key=api_key, model=request_model) as client:
			await client.aembed(text=texts_to_embed)

	assert "[openai] Authentication failed (Status: 401)" in str(excinfo.value)
	assert excinfo.value.provider == provider_name
	assert excinfo.value.status_code == 401


# Error handling tests for AsyncClient.aembed_batch (expecting BatchEmbeddingResponse with failed_indices)


@pytest.mark.asyncio
async def test_aembed_batch_authentication_error(httpx_mock):
	"""Test aembed_batch when API returns 401 for all batches.

	Expected: BatchEmbeddingResponse with all indices in failed_indices.
	"""
	provider_name = "openai"
	api_key = "invalid_key"
	request_model = "text-embedding-ada-002"
	texts_to_embed = ["text1", "text2"]  # Assumes this fits in one batch or multiple that all fail

	httpx_mock.add_response(
		url="https://api.openai.com/v1/embeddings",
		method="POST",
		status_code=401,
		json={"error": {"message": "Incorrect API key"}},
		# Match any batch payload since all will fail with 401
	)

	async with AsyncClient(provider=provider_name, api_key=api_key, model=request_model) as client:
		response = await client.aembed_batch(texts=texts_to_embed)

	assert isinstance(response, BatchEmbeddingResponse)
	assert len(response.embeddings) == 0
	assert response.failed_indices is not None
	assert len(response.failed_indices) == len(texts_to_embed)
	assert sorted(response.failed_indices) == sorted(range(len(texts_to_embed)))
	assert response.model == request_model
	assert response.provider == provider_name
	assert response.usage == {}  # No successful calls


@pytest.mark.asyncio
async def test_aembed_batch_rate_limit_error(httpx_mock):
	"""Test aembed_batch when API returns 429 for all batches.

	Expected: BatchEmbeddingResponse with all indices in failed_indices.
	"""
	provider_name = "openai"
	api_key = "fake_key"
	request_model = "text-embedding-ada-002"
	texts_to_embed = ["text1", "text2"]

	httpx_mock.add_response(
		url="https://api.openai.com/v1/embeddings",
		method="POST",
		status_code=429,
		json={"error": {"message": "Rate limit exceeded"}},
		headers={"retry-after": "60"},  # Header might be present but not used by BatchEmbeddingResponse here
	)

	async with AsyncClient(provider=provider_name, api_key=api_key, model=request_model) as client:
		response = await client.aembed_batch(texts=texts_to_embed)

	assert isinstance(response, BatchEmbeddingResponse)
	assert len(response.embeddings) == 0
	assert response.failed_indices is not None
	assert len(response.failed_indices) == len(texts_to_embed)
	assert sorted(response.failed_indices) == sorted(range(len(texts_to_embed)))
	assert response.model == request_model
	assert response.provider == provider_name
	assert response.usage == {}


@pytest.mark.asyncio
async def test_aembed_batch_invalid_request_error(httpx_mock):
	"""Test aembed_batch when API returns 400 for all batches.

	Expected: BatchEmbeddingResponse with all indices in failed_indices.
	"""
	provider_name = "openai"
	api_key = "fake_key"
	request_model = "text-embedding-ada-002"
	texts_to_embed = ["text1", "text2"]

	httpx_mock.add_response(
		url="https://api.openai.com/v1/embeddings",
		method="POST",
		status_code=400,
		json={"error": {"message": "Invalid request data"}},
	)

	async with AsyncClient(provider=provider_name, api_key=api_key, model=request_model) as client:
		response = await client.aembed_batch(texts=texts_to_embed)

	assert isinstance(response, BatchEmbeddingResponse)
	assert len(response.embeddings) == 0
	assert response.failed_indices is not None
	assert len(response.failed_indices) == len(texts_to_embed)
	assert sorted(response.failed_indices) == sorted(range(len(texts_to_embed)))
	assert response.model == request_model
	assert response.provider == provider_name
	assert response.usage == {}


@pytest.mark.asyncio
async def test_aembed_batch_generic_api_error(httpx_mock):
	"""Test aembed_batch when API returns 500 for all batches.

	Expected: BatchEmbeddingResponse with all indices in failed_indices.
	"""
	provider_name = "openai"
	api_key = "fake_key"
	request_model = "text-embedding-ada-002"
	texts_to_embed = ["text1", "text2"]

	httpx_mock.add_response(
		url="https://api.openai.com/v1/embeddings",
		method="POST",
		status_code=500,
		json={"error": {"message": "Internal server error"}},
	)

	async with AsyncClient(provider=provider_name, api_key=api_key, model=request_model) as client:
		response = await client.aembed_batch(texts=texts_to_embed)

	assert isinstance(response, BatchEmbeddingResponse)
	assert len(response.embeddings) == 0
	assert response.failed_indices is not None
	assert len(response.failed_indices) == len(texts_to_embed)
	assert sorted(response.failed_indices) == sorted(range(len(texts_to_embed)))
	assert response.model == request_model
	assert response.provider == provider_name
	assert response.usage == {}


@pytest.mark.asyncio
async def test_aembed_batch_non_embed_error_from_provider(httpx_mock, mocker):
	"""Test aembed_batch wraps non-EmbedError from provider client if that error is not an APIError.

	If the provider client's `aembed_batch` itself raises something like a ValueError
	(e.g., during its internal response processing, after a successful API call),
	the main client should wrap this in an EmbedError and correctly attribute
	failed indices.
	"""
	provider_name = "openai"
	api_key = "fake_key"
	request_model = "text-embedding-3-small"
	texts_to_embed = ["text_ok1", "problematic_text_that_causes_provider_value_error", "text_ok2"]

	# Mock the underlying provider's aembed_batch method
	# The goal is to have the *provider's* method raise a non-APIError for one of the batches.
	# Let's assume batch_size=1 for simplicity in mocking, so each text is a batch.
	# We need get_provider_client to return a mock that has a specially crafted aembed_batch.

	mock_provider_instance = mocker.AsyncMock(spec=OpenAIClient)  # Use a spec for safety

	async def mock_aembed_batch(request: BatchEmbeddingRequest, client: httpx.AsyncClient):
		if "problematic_text" in request.inputs[0]:  # Assuming batch_size 1 from main client
			msg = "Provider internal processing error for this specific text"
			raise ValueError(msg)

		# Successful response for other texts
		# Construct a simple, valid BatchEmbeddingResponse from the provider's perspective
		# (as if OpenAIClient.aembed_batch succeeded for this sub-batch)
		embeddings = [
			EmbeddingModel(values=[0.1, 0.2, 0.3], dimensions=3, index=0, text=request.inputs[0], model=request.model)
		]
		return BatchEmbeddingResponse(
			model=request.model,
			provider=provider_name,  # This would be OpenAI's view
			embeddings=embeddings,
			dimensions=3,
			texts=request.inputs,
			batch_count=1,
			failed_indices=None,
			usage={"prompt_tokens": 1, "total_tokens": 1},  # Dummy usage
		)

	mock_provider_instance.aembed_batch = mock_aembed_batch

	# Mock get_provider_client to return our mock_provider_instance
	mocker.patch("embed_all.client.async_client.get_provider_client", return_value=lambda _: mock_provider_instance)

	# Patch the batch size for the main AsyncClient to ensure individual processing for mock simplicity
	mocker.patch("embed_all.client.providers.open_ai.OPENAI_API_MAX_BATCH_SIZE", 1)

	async with AsyncClient(provider=provider_name, api_key=api_key, model=request_model) as client:
		# Set client's internal batch_size for the call if different from provider's max
		# Here, we rely on the patched OPENAI_API_MAX_BATCH_SIZE for client's internal splitting
		response = await client.aembed_batch(texts=texts_to_embed, batch_size=1)

	assert isinstance(response, BatchEmbeddingResponse)
	assert len(response.embeddings) == 2  # text_ok1 and text_ok2 should succeed
	assert response.failed_indices is not None
	assert response.failed_indices == [1]

	assert response.embeddings[0].values == [0.1, 0.2, 0.3]
	assert response.embeddings[0].text == texts_to_embed[0]
	assert response.embeddings[1].values == [0.1, 0.2, 0.3]  # Mock returns same embedding for simplicity
	assert response.embeddings[1].text == texts_to_embed[2]

	assert response.model == request_model
	assert response.provider == provider_name
	# Usage might be partial or reflect only successful calls, depending on implementation
	assert response.usage is not None
	# Total tokens should reflect the two successful calls (1+1 = 2)
	usage_data = cast("dict[str, Any]", response.usage)
	assert usage_data.get("total_tokens") == 2


@pytest.mark.parametrize("openai_api_max_batch_size_patch", [1])  # Force 1 text per API batch
@pytest.mark.asyncio
async def test_aembed_batch_partial_failure_multi_api_calls(httpx_mock, mocker, openai_api_max_batch_size_patch):
	"""Test aembed_batch with one internal API call failing and others succeeding."""
	mocker.patch("embed_all.client.providers.open_ai.OPENAI_API_MAX_BATCH_SIZE", openai_api_max_batch_size_patch)

	provider_name = "openai"
	api_key = "fake_api_key"
	request_model = "text-embedding-ada-002"
	texts_to_embed = ["text_success_batch1", "text_fail_batch2", "text_success_batch3"]

	json_batch1_success = {
		"object": "list",
		"data": [{"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0}],
		"model": request_model,
		"usage": {"prompt_tokens": 1, "total_tokens": 1},
	}
	json_batch3_success = {
		"object": "list",
		"data": [{"object": "embedding", "embedding": [0.7, 0.8, 0.9], "index": 0}],
		"model": request_model,
		"usage": {"prompt_tokens": 1, "total_tokens": 1},
	}

	httpx_mock.add_response(
		method="POST", url="https://api.openai.com/v1/embeddings", json=json_batch1_success, status_code=200
	)
	httpx_mock.add_response(
		method="POST",
		url="https://api.openai.com/v1/embeddings",
		json={"error": {"message": "This text is bad for batch 2"}},
		status_code=400,
	)
	httpx_mock.add_response(
		method="POST", url="https://api.openai.com/v1/embeddings", json=json_batch3_success, status_code=200
	)

	original_model_validate = OpenAIAPIResponse.model_validate
	mocker.patch(
		"embed_all.models.responses.providers.open_ai.OpenAIAPIResponse.model_validate",
		side_effect=lambda data, **kwargs: original_model_validate(data, **kwargs),
	)

	async with AsyncClient(provider=provider_name, api_key=api_key, model=request_model) as client:
		response = await client.aembed_batch(texts=texts_to_embed)

	assert isinstance(response, BatchEmbeddingResponse)
	assert len(response.embeddings) == 2
	assert response.failed_indices is not None
	assert len(response.failed_indices) == 1
	failed_list = list(response.failed_indices)
	assert 1 in failed_list

	assert response.embeddings[0].values == [0.1, 0.2, 0.3]
	assert response.embeddings[0].index == 0
	assert response.embeddings[1].values == [0.7, 0.8, 0.9]
	assert response.embeddings[1].index == 2

	assert response.model == request_model
	assert response.provider == provider_name
	assert response.usage is not None
	usage_data = cast("dict[str, Any]", response.usage)
	assert usage_data["prompt_tokens"] == 2
	assert usage_data["total_tokens"] == 2


@pytest.mark.asyncio
async def test_async_client_explicit_close_closes_http_client(mocker):
	"""Test that calling client.close() explicitly closes the HTTP client."""
	mock_async_http_client_instance = mocker.AsyncMock(spec=httpx.AsyncClient)
	mocker.patch("httpx.AsyncClient", return_value=mock_async_http_client_instance)

	client = AsyncClient(provider="openai", api_key="test_key")
	assert client._http_client == mock_async_http_client_instance

	await client.close()  # Call the public close method
	mock_async_http_client_instance.aclose.assert_called_once()

	# Verify that calling close again does not cause an error and aclose is still called once,
	# as AsyncClient should set _http_client to None after closing.
	await client.close()
	mock_async_http_client_instance.aclose.assert_called_once()  # Should still be 1


# Tests for synchronous wrappers on AsyncClient


def test_async_client_embed_wrapper_calls_aembed_and_warns(mocker, caplog):
	"""Test AsyncClient.embed calls AsyncClient.aembed and logs a warning."""
	provider_name = "openai"
	api_key = "test_key"
	text_to_embed = "hello from async embed wrapper"

	# Mock the underlying async aembed method
	mock_aembed = mocker.patch.object(AsyncClient, "aembed", autospec=True)
	# Create a mock response that aembed would return
	mock_embedding = EmbeddingModel(values=[0.1], dimensions=1, index=0, text=text_to_embed, model="test-model")
	mock_text_response = TextEmbeddingResponse(
		embeddings=[mock_embedding],
		dimensions=1,
		texts=[text_to_embed],
		model="test-model",
		provider=provider_name,
		usage={"prompt_tokens": 1, "total_tokens": 1},
	)
	mock_aembed.return_value = (
		mock_text_response  # aembed is async, so its mock should behave like an async func if awaited
	)
	# but here, the wrapper will call it and handle the event loop.

	# We need an instance of AsyncClient to call the sync wrapper on.
	# The HTTP client aspects don't matter here as we are mocking aembed directly.
	client = AsyncClient(provider=provider_name, api_key=api_key)

	caplog.set_level(logging.WARNING)  # Ensure WARNING level is captured
	response = client.embed(text=text_to_embed, model="test-model")  # Call the sync wrapper

	mock_aembed.assert_called_once_with(
		client,
		text=text_to_embed,
		model="test-model",
		dimensions=None,  # Add other defaults as needed
	)
	assert response == mock_text_response
	assert len(caplog.records) == 1
	assert caplog.records[0].levelname == "WARNING"
	assert "Using async client with sync method. For better control, use aembed for async operations." in caplog.text

	# Clean up the client if it manages its own loop that needs closing, though mocking aembed might bypass this.
	# No explicit client.close() here as the test is about the embed method, not lifecycle


def test_async_client_embed_batch_wrapper_calls_aembed_batch_and_warns(mocker, caplog):
	"""Test AsyncClient.embed_batch calls AsyncClient.aembed_batch and logs a warning."""
	provider_name = "openai"
	api_key = "test_key"
	texts_to_embed = ["text1", "text2"]

	# Mock the underlying async aembed_batch method
	mock_aembed_batch = mocker.patch.object(AsyncClient, "aembed_batch", autospec=True)
	mock_embedding1 = EmbeddingModel(values=[0.1], dimensions=1, index=0, text=texts_to_embed[0], model="test-model")
	mock_embedding2 = EmbeddingModel(values=[0.2], dimensions=1, index=1, text=texts_to_embed[1], model="test-model")
	mock_batch_response = BatchEmbeddingResponse(
		embeddings=[mock_embedding1, mock_embedding2],
		dimensions=1,
		texts=texts_to_embed,
		batch_count=1,
		failed_indices=None,
		model="test-model",
		provider=provider_name,
		usage={"prompt_tokens": 2, "total_tokens": 2},
	)
	mock_aembed_batch.return_value = mock_batch_response

	client = AsyncClient(provider=provider_name, api_key=api_key)

	caplog.set_level(logging.WARNING)
	response = client.embed_batch(texts=texts_to_embed, model="test-model")  # Call the sync wrapper

	# Assert that the underlying aembed_batch was called correctly
	mock_aembed_batch.assert_called_once_with(
		client,
		texts=texts_to_embed,
		model="test-model",
		dimensions=None,
		batch_size=32,  # Default batch_size
	)
	assert response == mock_batch_response
	assert len(caplog.records) == 1
	assert caplog.records[0].levelname == "WARNING"
	assert (
		"Using async client with sync method. For better control, use aembed_batch for async operations." in caplog.text
	)

	# Clean up client if necessary
	# No explicit client.close() for same reasons as above
