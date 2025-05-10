"""Tests for embed_all.client.sync_client module."""

from typing import Any, cast

import httpx
import pytest

from embed_all.client.sync_client import Client
from embed_all.models.base import EmbeddingModel
from embed_all.models.errors import APIError, AuthenticationError, EmbedError, InvalidRequestError, RateLimitError
from embed_all.models.responses.batch import BatchEmbeddingResponse
from embed_all.models.responses.providers.open_ai import OpenAIAPIResponse
from embed_all.models.responses.text import TextEmbeddingResponse


def test_sync_client_initialization():
	"""Test that Client initializes correctly and resolves default model."""
	# Assuming "openai" and its default model for this test, similar to AsyncClient test
	client = Client(provider="openai", api_key="test_key")
	assert client.config.provider == "openai"
	assert client.config.api_key == "test_key"
	# BaseClient.__init__ ensures config.model is the resolved default model string
	assert client.config.model == "text-embedding-3-small"  # Expected default for openai


def test_embed_simple_text(httpx_mock):
	"""Test embed with a single text string - happy path."""
	provider_name = "openai"
	api_key = "fake_api_key"
	request_model = "text-embedding-ada-002"

	mock_response_data = {
		"object": "list",
		"data": [
			{
				"object": "embedding",
				"embedding": [0.7, 0.8, 0.9],
				"index": 0,
			}
		],
		"model": request_model,
		"usage": {"prompt_tokens": 6, "total_tokens": 6},
	}
	httpx_mock.add_response(
		url="https://api.openai.com/v1/embeddings",
		method="POST",
		json=mock_response_data,
		status_code=200,
	)

	with Client(provider=provider_name, api_key=api_key, model=request_model) as client:
		response = client.embed(text="Hello, sync world!")

	assert isinstance(response, TextEmbeddingResponse)
	assert len(response.embeddings) == 1
	assert isinstance(response.embeddings[0], EmbeddingModel)
	assert response.embeddings[0].values == [0.7, 0.8, 0.9]
	assert response.model == request_model
	assert response.provider == provider_name
	assert response.usage is not None
	usage_data = cast("dict[str, Any]", response.usage)
	assert usage_data["prompt_tokens"] == 6
	assert usage_data["total_tokens"] == 6


# Error handling tests for Client.embed


def test_embed_authentication_error(httpx_mock):
	"""Test embed raises AuthenticationError on 401."""
	provider_name = "openai"
	api_key = "invalid_key"
	request_model = "text-embedding-ada-002"
	texts_to_embed = ["Hello, world!"]

	httpx_mock.add_response(
		url="https://api.openai.com/v1/embeddings",
		method="POST",
		status_code=401,
		json={"error": {"message": "Incorrect API key"}},
	)

	with (
		pytest.raises(AuthenticationError) as excinfo,
		Client(provider=provider_name, api_key=api_key, model=request_model) as client,
	):
		client.embed(text=texts_to_embed)

	assert "[openai] Authentication failed (Status: 401)" in str(excinfo.value)
	assert excinfo.value.provider == provider_name
	assert excinfo.value.status_code == 401


def test_embed_rate_limit_error(httpx_mock):
	"""Test embed raises RateLimitError on 429."""
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

	with (
		pytest.raises(RateLimitError) as excinfo,
		Client(provider=provider_name, api_key=api_key, model=request_model) as client,
	):
		client.embed(text="Hello, world!")

	assert "[openai] Rate limit exceeded (Status: 429)" in str(excinfo.value)
	assert excinfo.value.provider == provider_name
	assert excinfo.value.status_code == 429
	assert excinfo.value.retry_after == 60


def test_embed_invalid_request_error(httpx_mock):
	"""Test embed raises InvalidRequestError on 400."""
	provider_name = "openai"
	api_key = "fake_key"
	request_model = "text-embedding-ada-002"

	httpx_mock.add_response(
		url="https://api.openai.com/v1/embeddings",
		method="POST",
		status_code=400,
		json={"error": {"message": "Invalid request data"}},
	)

	with (
		pytest.raises(InvalidRequestError) as excinfo,
		Client(provider=provider_name, api_key=api_key, model=request_model) as client,
	):
		client.embed(text="Hello, world!")

	assert "[openai] Invalid request data (Status: 400)" in str(excinfo.value)
	assert excinfo.value.provider == provider_name
	assert excinfo.value.status_code == 400


def test_embed_generic_api_error(httpx_mock):
	"""Test embed raises APIError on other HTTP errors (e.g., 500)."""
	provider_name = "openai"
	api_key = "fake_key"
	request_model = "text-embedding-ada-002"

	httpx_mock.add_response(
		url="https://api.openai.com/v1/embeddings",
		method="POST",
		status_code=500,
		json={"error": {"message": "Internal server error"}},
	)

	with (
		pytest.raises(APIError) as excinfo,
		Client(provider=provider_name, api_key=api_key, model=request_model) as client,
	):
		client.embed(text="Hello, world!")

	assert "[openai] Internal server error (Status: 500)" in str(excinfo.value)
	assert excinfo.value.provider == provider_name
	assert excinfo.value.status_code == 500
	assert not isinstance(excinfo.value, AuthenticationError)
	assert not isinstance(excinfo.value, RateLimitError)
	assert not isinstance(excinfo.value, InvalidRequestError)


def test_embed_non_embed_error_from_provider(mocker):
	"""Test embed wraps non-EmbedError from provider into EmbedError."""
	provider_name = "openai"
	api_key = "fake_key"
	request_model = "text-embedding-3-small"

	# Mock the provider client's embed to raise a generic Exception
	mock_provider_client = mocker.MagicMock()
	mock_provider_client.embed.side_effect = ValueError("Provider internal error")

	# Mock get_provider_client to return our mock_provider_client
	# when Client initializes it.
	mocker.patch("embed_all.client.sync_client.get_provider_client", return_value=lambda _: mock_provider_client)

	with (
		Client(provider=provider_name, api_key=api_key, model=request_model) as client,
		pytest.raises(EmbedError) as excinfo,
	):
		client.embed(text="Hello, error world!")

	assert "Error generating embeddings: Provider internal error" in str(excinfo.value)
	assert excinfo.value.provider == provider_name
	assert not isinstance(excinfo.value, (AuthenticationError, RateLimitError, InvalidRequestError))
	# Ensure the original error is chained
	assert isinstance(excinfo.value.__cause__, ValueError)
	assert str(excinfo.value.__cause__) == "Provider internal error"


def test_embed_batch_simple(httpx_mock):
	"""Test embed_batch with a list of texts - happy path."""
	provider_name = "openai"
	api_key = "fake_api_key"
	request_model = "text-embedding-ada-002"
	texts_to_embed = ["Hello, batch world!", "Another text for batch."]

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
		"usage": {"prompt_tokens": 12, "total_tokens": 12},
	}
	# Assuming OpenAI provider which might make one call for a small batch
	httpx_mock.add_response(
		url="https://api.openai.com/v1/embeddings",
		method="POST",
		json=mock_response_data,
		status_code=200,
	)

	with Client(provider=provider_name, api_key=api_key, model=request_model) as client:
		response = client.embed_batch(texts=texts_to_embed)

	assert isinstance(response, BatchEmbeddingResponse)
	assert len(response.embeddings) == len(texts_to_embed)
	assert len(response.embeddings) == len(texts_to_embed)  # Assuming all succeed for happy path
	# Check for no failed indices
	if response.failed_indices is not None:
		# This path should ideally not be taken in a happy path test if None is expected
		# but if it can be an empty list, this handles it.
		failed_list = cast("list[int]", response.failed_indices)
		assert len(failed_list) == 0
	# else: response.failed_indices is None, which is also success.

	assert response.model == request_model
	assert response.provider == provider_name
	assert response.usage is not None
	usage_data = cast("dict[str, Any]", response.usage)
	assert usage_data["prompt_tokens"] == 12
	assert usage_data["total_tokens"] == 12

	for i, embedding in enumerate(response.embeddings):
		assert isinstance(embedding, EmbeddingModel)
		assert embedding.index == i
		# Actual values depend on the mock response
		if i == 0:
			assert embedding.values == [0.1, 0.2, 0.3]
		elif i == 1:
			assert embedding.values == [0.4, 0.5, 0.6]


# Error handling tests for Client.embed_batch (expecting BatchEmbeddingResponse with failed_indices)


def test_embed_batch_authentication_error(httpx_mock):
	"""Test embed_batch when API returns 401 for all batches.

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

	with Client(provider=provider_name, api_key=api_key, model=request_model) as client:
		response = client.embed_batch(texts=texts_to_embed)

	assert isinstance(response, BatchEmbeddingResponse)
	assert len(response.embeddings) == 0
	assert response.failed_indices is not None
	failed_list = cast("list[int]", response.failed_indices)
	assert len(failed_list) == len(texts_to_embed)
	assert failed_list == list(range(len(texts_to_embed)))
	assert response.model == request_model
	assert response.provider == provider_name
	assert response.usage == {}  # No successful calls


def test_embed_batch_rate_limit_error(httpx_mock):
	"""Test embed_batch when API returns 429 for all batches.

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

	with Client(provider=provider_name, api_key=api_key, model=request_model) as client:
		response = client.embed_batch(texts=texts_to_embed)

	assert isinstance(response, BatchEmbeddingResponse)
	assert len(response.embeddings) == 0
	assert response.failed_indices is not None
	failed_list = cast("list[int]", response.failed_indices)
	assert len(failed_list) == len(texts_to_embed)
	assert failed_list == list(range(len(texts_to_embed)))
	assert response.model == request_model
	assert response.provider == provider_name
	assert response.usage == {}
	# Note: BatchEmbeddingResponse itself doesn't have a retry_after field.
	# The RateLimitError exception (if it were raised) would.


def test_embed_batch_invalid_request_error(httpx_mock):
	"""Test embed_batch when API returns 400 for all batches.

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

	with Client(provider=provider_name, api_key=api_key, model=request_model) as client:
		response = client.embed_batch(texts=texts_to_embed)

	assert isinstance(response, BatchEmbeddingResponse)
	assert len(response.embeddings) == 0
	assert response.failed_indices is not None
	failed_list = cast("list[int]", response.failed_indices)
	assert len(failed_list) == len(texts_to_embed)
	assert failed_list == list(range(len(texts_to_embed)))
	assert response.model == request_model
	assert response.provider == provider_name
	assert response.usage == {}


def test_embed_batch_generic_api_error(httpx_mock):
	"""Test embed_batch when API returns 500 for all batches.

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

	with Client(provider=provider_name, api_key=api_key, model=request_model) as client:
		response = client.embed_batch(texts=texts_to_embed)

	assert isinstance(response, BatchEmbeddingResponse)
	assert len(response.embeddings) == 0
	assert response.failed_indices is not None
	failed_list = cast("list[int]", response.failed_indices)
	assert len(failed_list) == len(texts_to_embed)
	assert failed_list == list(range(len(texts_to_embed)))
	assert response.model == request_model
	assert response.provider == provider_name
	assert response.usage == {}


@pytest.mark.parametrize("openai_api_max_batch_size_patch", [1])  # Force 1 text per API batch
def test_embed_batch_partial_failure_multi_api_calls(httpx_mock, mocker, openai_api_max_batch_size_patch):
	"""Test embed_batch with partial success where texts span multiple API batches.

	One API batch succeeds, another fails.
	"""
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
	# json_batch2_fail_content will be handled by HTTP status code 400 from httpx_mock
	json_batch3_success = {
		"object": "list",
		"data": [{"object": "embedding", "embedding": [0.7, 0.8, 0.9], "index": 0}],
		"model": request_model,
		"usage": {"prompt_tokens": 1, "total_tokens": 1},
	}

	# Mock HTTP responses
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

	# Ensure OpenAIAPIResponse.model_validate behaves normally for this test, in case of mock leakage
	# by explicitly making it call the original for the successful calls.
	# The failing call (batch 2) won't reach model_validate due to raise_for_status().
	original_model_validate = OpenAIAPIResponse.model_validate
	mocker.patch(
		"embed_all.models.responses.providers.open_ai.OpenAIAPIResponse.model_validate",
		side_effect=lambda data: original_model_validate(data),  # Always use original for this test's scope
	)

	with Client(provider=provider_name, api_key=api_key, model=request_model) as client:
		response = client.embed_batch(texts=texts_to_embed)

	assert isinstance(response, BatchEmbeddingResponse)
	assert len(response.embeddings) == 2  # First and third text succeeded
	assert response.failed_indices is not None
	failed_list = cast("list[int]", response.failed_indices)
	assert len(failed_list) == 1
	assert 1 in failed_list  # Index 1 (text_fail_batch2) should have failed

	assert response.embeddings[0].values == [0.1, 0.2, 0.3]
	assert response.embeddings[0].index == 0  # Original index of text_success_batch1
	assert response.embeddings[1].values == [0.7, 0.8, 0.9]
	assert response.embeddings[1].index == 2  # Original index of text_success_batch3

	assert response.model == request_model
	assert response.provider == provider_name
	assert response.usage is not None
	usage_data = cast("dict[str, Any]", response.usage)
	assert usage_data["prompt_tokens"] == 2
	assert usage_data["total_tokens"] == 2


def test_embed_batch_all_fail_auth_error(httpx_mock):
	"""
	Test embed_batch when all underlying API calls fail with an auth error.

	Expected: Returns BatchEmbeddingResponse with all indices marked as failed,
	does not raise AuthenticationError from Client.embed_batch itself.
	"""
	provider_name = "openai"
	api_key = "invalid_key"
	request_model = "text-embedding-ada-002"
	# Assume OPENAI_API_MAX_BATCH_SIZE is small enough that this requires multiple batches,
	# or test with a single batch that fails.
	# For simplicity, let's use a number of texts that fits in one default batch (e.g., 2)
	# but the test structure should hold even if OpenAIClient internally makes multiple calls.
	texts_to_embed = ["text1", "text2"]

	# Mock API to always return 401
	httpx_mock.add_response(
		url="https://api.openai.com/v1/embeddings",
		method="POST",
		status_code=401,
		json={"error": {"message": "Incorrect API key"}},
	)

	with Client(provider=provider_name, api_key=api_key, model=request_model) as client:
		response = client.embed_batch(texts=texts_to_embed)

	assert isinstance(response, BatchEmbeddingResponse)
	assert len(response.embeddings) == 0  # No successful embeddings
	assert response.failed_indices is not None
	failed_list = cast("list[int]", response.failed_indices)
	assert len(failed_list) == len(texts_to_embed)
	assert failed_list == list(range(len(texts_to_embed)))
	assert response.model == request_model
	assert response.provider == provider_name
	assert response.usage == {}  # No successful calls, so usage should be empty


def test_embed_batch_non_embed_error_from_provider(httpx_mock, mocker):
	"""Test embed_batch when provider logic raises an unexpected non-EmbedError.

	Expected: BatchEmbeddingResponse with relevant indices in failed_indices.
	The original error should be logged by the provider but not raised from Client.embed_batch.
	"""
	provider_name = "openai"
	api_key = "fake_api_key"
	request_model = "text-embedding-ada-002"
	texts_to_embed = ["text1", "problematic_text_causes_internal_error", "text3"]

	# We'll mock three batches (OPENAI_API_MAX_BATCH_SIZE = 1 for this test)
	mocker.patch("embed_all.client.providers.open_ai.OPENAI_API_MAX_BATCH_SIZE", 1)

	json_batch1 = {
		"object": "list",
		"data": [{"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0}],
		"model": request_model,
		"usage": {"prompt_tokens": 1, "total_tokens": 1},
	}
	json_batch2_content = {  # Content for the API call that will have its validation mocked to fail
		"object": "list",
		"data": [{"object": "embedding", "embedding": [0.4, 0.5, 0.6], "index": 0}],
		"model": request_model,
		"usage": {"prompt_tokens": 1, "total_tokens": 1},
	}
	json_batch3 = {
		"object": "list",
		"data": [{"object": "embedding", "embedding": [0.7, 0.8, 0.9], "index": 0}],
		"model": request_model,
		"usage": {"prompt_tokens": 1, "total_tokens": 1},
	}

	# Batch 1: Success
	httpx_mock.add_response(
		url="https://api.openai.com/v1/embeddings", method="POST", json=json_batch1, status_code=200
	)

	# Batch 2: API Success, but model_validate will fail
	httpx_mock.add_response(
		url="https://api.openai.com/v1/embeddings", method="POST", json=json_batch2_content, status_code=200
	)

	# Batch 3: Success
	httpx_mock.add_response(
		url="https://api.openai.com/v1/embeddings", method="POST", json=json_batch3, status_code=200
	)

	mocker.patch(
		"embed_all.models.responses.providers.open_ai.OpenAIAPIResponse.model_validate",
		side_effect=[
			OpenAIAPIResponse.model_validate(json_batch1),  # Batch 1: Success, pre-validated
			ValueError("Unexpected internal validation error"),  # Batch 2: Raise error
			OpenAIAPIResponse.model_validate(json_batch3),  # Batch 3: Success, pre-validated
		],
	)

	with Client(provider=provider_name, api_key=api_key, model=request_model) as client:
		response = client.embed_batch(texts=texts_to_embed)

	assert isinstance(response, BatchEmbeddingResponse)
	assert len(response.embeddings) == 2  # Batch 1 and 3 succeeded
	assert response.failed_indices is not None
	failed_list = cast("list[int]", response.failed_indices)
	assert len(failed_list) == 1
	assert 1 in failed_list  # Index 1 (problematic_text_...) failed due to internal error

	assert response.embeddings[0].values == [0.1, 0.2, 0.3]
	assert response.embeddings[0].index == 0
	assert response.embeddings[1].values == [0.7, 0.8, 0.9]
	assert response.embeddings[1].index == 2

	assert response.model == request_model
	assert response.provider == provider_name
	assert response.usage is not None
	usage_data = cast("dict[str, Any]", response.usage)
	assert usage_data["prompt_tokens"] == 2  # For batch 1 and 3
	assert usage_data["total_tokens"] == 2

	# We expect the provider to log the ValueError, but not for Client to raise it.
	# Checking log records would be more involved here, relying on failed_indices is sufficient for Client behavior.


def test_client_context_manager_closes_http_client(mocker):
	"""Test that the Client's HTTP client is closed when used as a context manager."""
	provider_name = "openai"
	api_key = "test_key"

	mock_http_client = mocker.MagicMock(spec=httpx.Client)

	# Mock the __init__ of httpx.Client to return our mock_http_client
	# This way, when BaseClient creates its self._http_client, it gets our mock.
	mock_httpx_client_constructor = mocker.patch("httpx.Client", return_value=mock_http_client)

	with Client(provider=provider_name, api_key=api_key) as client:
		assert client._http_client == mock_http_client
		mock_http_client.close.assert_not_called()  # Should not be closed yet

	# After exiting the context manager, close should have been called
	mock_http_client.close.assert_called_once()
	# Ensure httpx.Client was actually instantiated by BaseClient
	mock_httpx_client_constructor.assert_called_once()


def test_client_explicit_close_closes_http_client(mocker):
	"""Test that calling client.close() explicitly closes the HTTP client."""
	provider_name = "openai"
	api_key = "test_key"

	mock_http_client = mocker.MagicMock(spec=httpx.Client)
	mock_httpx_client_constructor = mocker.patch("httpx.Client", return_value=mock_http_client)

	client = Client(provider=provider_name, api_key=api_key)
	assert client._http_client == mock_http_client

	client.close()
	mock_http_client.close.assert_called_once()
	mock_httpx_client_constructor.assert_called_once()


# Tests for Asynchronous Wrappers in Synchronous Client


@pytest.mark.asyncio
async def test_sync_client_aembed_wrapper(mocker, httpx_mock):
	"""Test that SyncClient.aembed calls SyncClient.embed and logs a warning."""
	provider_name = "openai"
	api_key = "test_key"
	text_to_embed = "hello from sync aembed wrapper"

	# Mock the underlying sync embed method
	mock_embed = mocker.patch.object(Client, "embed", autospec=True)
	mock_embedding = EmbeddingModel(values=[0.1], dimensions=1, index=0, text=text_to_embed, model="test-model")
	mock_text_response = TextEmbeddingResponse(
		embeddings=[mock_embedding],
		dimensions=1,
		texts=[text_to_embed],
		model="test-model",
		provider=provider_name,
		usage={"prompt_tokens": 1, "total_tokens": 1},
	)
	mock_embed.return_value = mock_text_response

	client = Client(provider=provider_name, api_key=api_key)

	with pytest.warns(
		UserWarning,
		match="Running asynchronous `aembed` from synchronous Client. "
		"For true non-blocking async behavior, use `AsyncClient`.",
	) as record:
		response = await client.aembed(text=text_to_embed, model="test-model")

	mock_embed.assert_called_once_with(client, text=text_to_embed, model="test-model", dimensions=None)
	assert response == mock_text_response
	assert len(record) == 1


@pytest.mark.asyncio
async def test_sync_client_aembed_batch_wrapper(mocker, httpx_mock):
	"""Test that SyncClient.aembed_batch calls SyncClient.embed_batch and logs a warning."""
	provider_name = "openai"
	api_key = "test_key"
	texts_to_embed = ["text1", "text2"]

	# Mock the underlying sync embed_batch method
	mock_embed_batch = mocker.patch.object(Client, "embed_batch", autospec=True)
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
	mock_embed_batch.return_value = mock_batch_response

	client = Client(provider=provider_name, api_key=api_key)

	with pytest.warns(
		UserWarning,
		match="Running asynchronous `aembed_batch` from synchronous Client. "
		"For true non-blocking async behavior, use `AsyncClient`.",
	) as record:
		response = await client.aembed_batch(texts=texts_to_embed, model="test-model")

	mock_embed_batch.assert_called_once_with(
		client, texts=texts_to_embed, model="test-model", dimensions=None, batch_size=32
	)
	assert response == mock_batch_response
	assert len(record) == 1


def test_embed_list_input_happy_path(httpx_mock):
	"""Test embed with a list of text strings - happy path."""
	provider_name = "openai"
	api_key = "fake_api_key"
	request_model = "text-embedding-ada-002"
	texts_to_embed = ["Hello, sync list world!", "Another text for sync list."]

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
		"usage": {"prompt_tokens": 12, "total_tokens": 12},  # Example usage
	}
	httpx_mock.add_response(
		url="https://api.openai.com/v1/embeddings",
		method="POST",
		json=mock_response_data,
		status_code=200,
	)

	with Client(provider=provider_name, api_key=api_key, model=request_model) as client:
		response = client.embed(text=texts_to_embed)

	assert isinstance(response, TextEmbeddingResponse)
	assert len(response.embeddings) == len(texts_to_embed)
	assert response.texts == texts_to_embed
	assert response.model == request_model
	assert response.provider == provider_name
	assert response.usage is not None
	usage_data = cast("dict[str, Any]", response.usage)
	assert usage_data["prompt_tokens"] == 12
	assert usage_data["total_tokens"] == 12

	for i, embedding in enumerate(response.embeddings):
		assert isinstance(embedding, EmbeddingModel)
		assert embedding.index == i
		if i == 0:
			assert embedding.values == [0.1, 0.2, 0.3]
		elif i == 1:
			assert embedding.values == [0.4, 0.5, 0.6]


def test_embed_list_input_authentication_error(httpx_mock):
	"""Test embed with a list of text strings raises AuthenticationError."""
	provider_name = "openai"
	api_key = "invalid_key"
	request_model = "text-embedding-ada-002"
	texts_to_embed = ["text1_sync_list_err", "text2_sync_list_err"]

	httpx_mock.add_response(
		url="https://api.openai.com/v1/embeddings",
		method="POST",
		status_code=401,
		json={"error": {"message": "Incorrect API key for list input"}},
	)

	with (
		pytest.raises(AuthenticationError) as excinfo,
		Client(provider=provider_name, api_key=api_key, model=request_model) as client,
	):
		client.embed(text=texts_to_embed)

	assert "[openai] Authentication failed (Status: 401)" in str(excinfo.value)
	assert excinfo.value.provider == provider_name
	assert excinfo.value.status_code == 401
