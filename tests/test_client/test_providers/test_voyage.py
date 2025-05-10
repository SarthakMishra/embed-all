"""Tests for src/embed_all/client/providers/voyage.py."""

from typing import cast

import httpx
import pytest

from embed_all.client.providers import voyage as voyage_provider
from embed_all.client.providers.voyage import VoyageClient
from embed_all.models import (
	BatchEmbeddingRequest,
	BatchEmbeddingResponse,
	ProviderModel,
	TextEmbeddingRequest,
	TextEmbeddingResponse,
)
from embed_all.models.errors import AuthenticationError
from embed_all.models.responses.providers.voyage import VoyageAPIResponse, VoyageEmbeddingData, VoyageUsage


def test_voyage_client_initialization_requires_api_key():
	"""Test VoyageClient initialization raises AuthenticationError if api_key is not provided."""
	with pytest.raises(AuthenticationError, match=r"^\[voyage\] API key is required for Voyage AI$") as excinfo:
		VoyageClient(
			config=ProviderModel(
				provider="voyage",
				api_key=None,
				base_url=None,
				model=None,
				timeout=60,
				max_retries=3,
			)
		)
	assert "[voyage] API key is required for Voyage AI" in str(excinfo.value)
	assert excinfo.value.provider == "voyage"


def test_voyage_client_initialization_with_api_key():
	"""Test VoyageClient initialization succeeds with an API key."""
	try:
		VoyageClient(
			config=ProviderModel(
				provider="voyage",
				api_key="test_voyage_key",
				base_url=None,
				model=None,
				timeout=60,
				max_retries=3,
			)
		)
	except AuthenticationError:
		pytest.fail("VoyageClient initialization failed with a valid API key and should raise AuthenticationError")
	except Exception as e:
		pytest.fail(f"VoyageClient initialization failed unexpectedly: {e}")


def test_voyage_client_get_headers():
	"""Test VoyageClient._get_headers returns the correct headers."""
	api_key = "test_voyage_api_key"
	client = VoyageClient(
		config=ProviderModel(
			provider="voyage",
			api_key=api_key,
			base_url=None,
			model=None,
			timeout=60,
			max_retries=3,
		)
	)
	headers = client._get_headers()

	assert headers["Authorization"] == f"Bearer {api_key}"
	assert headers["Content-Type"] == "application/json"
	assert len(headers) == 2


def test_voyage_client_process_embeddings_response_valid():
	"""Test VoyageClient._process_embeddings_response with a valid API JSON response."""
	api_key = "test_voyage_key"
	client = VoyageClient(
		config=ProviderModel(
			provider="voyage",
			api_key=api_key,
			model="voyage-2",
			base_url=None,
			timeout=60,
			max_retries=3,
		)
	)
	request_model_name = "voyage-2"
	input_texts: list[str] = ["Hello Voyage!", "Another Voyage text."]

	# Mock Voyage API JSON response structure
	# The VoyageAPIResponse model does not have a top-level "model" field.
	# It expects a list of VoyageEmbeddingObject for "data".
	# Let's adjust the mock and validation.
	validated_api_response = VoyageAPIResponse(
		object="list",
		data=[
			VoyageEmbeddingData(object="embedding", embedding=[0.1, 0.2, 0.3, 0.4], index=0),
			VoyageEmbeddingData(object="embedding", embedding=[0.5, 0.6, 0.7, 0.8], index=1),
		],
		model=request_model_name,
		usage=VoyageUsage(total_tokens=10),
	)

	processed_response = client._process_embeddings_response(validated_api_response, input_texts, request_model_name)

	assert isinstance(processed_response, TextEmbeddingResponse)
	assert len(processed_response.embeddings) == 2
	assert processed_response.texts == input_texts
	assert processed_response.model == request_model_name
	assert processed_response.provider == "voyage"
	assert processed_response.usage == {"total_tokens": 10}
	assert processed_response.embeddings[0].values == [0.1, 0.2, 0.3, 0.4]
	# Voyage API doesn't provide an index per embedding in the response, so we assign it based on order.
	assert processed_response.embeddings[0].index == 0
	assert processed_response.embeddings[1].values == [0.5, 0.6, 0.7, 0.8]
	assert processed_response.embeddings[1].index == 1


def test_voyage_client_process_embeddings_response_empty_data():
	"""Test VoyageClient._process_embeddings_response with an API JSON response containing an empty 'data' list."""
	api_key = "test_voyage_key_empty"
	client = VoyageClient(
		config=ProviderModel(
			provider="voyage",
			api_key=api_key,
			model="voyage-large-2",
			base_url=None,
			timeout=60,
			max_retries=3,
		)
	)
	request_model_name = "voyage-large-2"
	input_texts: list[str] = []

	validated_api_response = VoyageAPIResponse(
		object="list", data=[], model=request_model_name, usage=VoyageUsage(total_tokens=0)
	)

	# Expect an empty TextEmbeddingResponse, not an APIError
	processed_response = client._process_embeddings_response(validated_api_response, input_texts, request_model_name)

	assert isinstance(processed_response, TextEmbeddingResponse)
	assert len(processed_response.embeddings) == 0
	assert processed_response.texts == input_texts  # Should be an empty list
	assert processed_response.model == request_model_name
	assert processed_response.provider == "voyage"
	assert processed_response.dimensions == 0  # As set in _process_embeddings_response for empty data
	assert processed_response.usage == {"total_tokens": 0}


# Tests for embed/aembed direct calls for VoyageClient
# import httpx # Already imported at the top


@pytest.mark.asyncio
async def test_voyage_client_aembed_happy_path(httpx_mock):
	"""Test a basic successful call to VoyageClient.aembed."""
	api_key = "test_key_voyage_aembed"
	model_name = "voyage-2"  # Example Voyage model
	provider_config = ProviderModel(
		provider="voyage", api_key=api_key, model=model_name, base_url=None, timeout=60, max_retries=3
	)
	voyage_client = VoyageClient(config=provider_config)

	input_text = "Test input for VoyageClient.aembed"
	request_data = TextEmbeddingRequest(
		model=model_name,
		input=input_text,
		dimensions=None,  # Explicitly None
		encoding_format=None,  # Explicitly None
		input_type=None,  # Explicitly None
		truncate=None,  # Explicitly None
	)

	# Mock Voyage API response. Note: Voyage's actual API might differ.
	# This mock should align with how VoyageClient._process_embeddings_response expects it,
	# which uses VoyageAPIResponse and VoyageEmbeddingData.
	mock_api_response_payload = {
		"object": "list",
		"data": [
			# VoyageEmbeddingData expects 'index' from the API response as per its Pydantic model.
			# However, the actual Voyage API might not return 'index' in each embedding object.
			# Let's assume _process_embeddings_response handles this (e.g., by using list order).
			# For this direct test, we construct what _process_embeddings_response gets AFTER API validation.
			{"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0}
		],
		"model": model_name,
		"usage": {"total_tokens": 7},
	}
	httpx_mock.add_response(
		url=f"{voyage_client.base_url}/embeddings",
		method="POST",
		json=mock_api_response_payload,
		status_code=200,
	)

	async with httpx.AsyncClient() as http_client:
		response = await voyage_client.aembed(request_data, http_client)

	assert isinstance(response, TextEmbeddingResponse)
	assert len(response.embeddings) == 1
	assert response.embeddings[0].values == [0.1, 0.2, 0.3]
	assert response.model == model_name
	assert response.texts == [input_text]
	assert response.usage == {"total_tokens": 7}


def test_voyage_client_embed_happy_path(httpx_mock):
	"""Test a basic successful call to VoyageClient.embed."""
	api_key = "test_key_voyage_embed"
	model_name = "voyage-lite-02-instruct"  # Another example Voyage model
	provider_config = ProviderModel(
		provider="voyage", api_key=api_key, model=model_name, base_url=None, timeout=60, max_retries=3
	)
	voyage_client = VoyageClient(config=provider_config)

	input_text = "Test input for VoyageClient.embed"
	request_data = TextEmbeddingRequest(
		model=model_name,
		input=input_text,
		dimensions=None,  # Explicitly None
		encoding_format=None,  # Explicitly None
		input_type=None,  # Explicitly None
		truncate=None,  # Explicitly None
	)

	mock_api_response_payload = {
		"object": "list",
		"data": [{"object": "embedding", "embedding": [0.6, 0.7, 0.8], "index": 0}],
		"model": model_name,
		"usage": {"total_tokens": 8},
	}
	httpx_mock.add_response(
		url=f"{voyage_client.base_url}/embeddings",
		method="POST",
		json=mock_api_response_payload,
		status_code=200,
	)

	with httpx.Client() as http_client:
		response = voyage_client.embed(request_data, http_client)

	assert isinstance(response, TextEmbeddingResponse)
	assert len(response.embeddings) == 1
	assert response.embeddings[0].values == [0.6, 0.7, 0.8]
	assert response.model == model_name
	assert response.texts == [input_text]
	assert response.usage == {"total_tokens": 8}


# Tests for embed_batch/aembed_batch direct calls (Voyage-specific batching)

# VOYAGE_API_MAX_BATCH_SIZE is 128 (from their docs), but we'll use a smaller number for testing ease
# or mock the constant `VOYAGE_INPUT_BATCH_SIZE` in `voyage.py` if necessary.


@pytest.mark.asyncio
async def test_voyage_client_aembed_batch_less_than_max_size(httpx_mock):
	"""Test VoyageClient.aembed_batch with input size < VOYAGE_INPUT_BATCH_SIZE."""
	api_key = "test_voyage_aembed_batch_small"
	model_name = "voyage-2"
	provider_config = ProviderModel(
		provider="voyage", api_key=api_key, model=model_name, base_url=None, timeout=60, max_retries=3
	)
	voyage_client = VoyageClient(config=provider_config)

	input_texts = ["voyage batch text 1", "voyage batch text 2"]  # 2 items, less than default 7
	request_data = BatchEmbeddingRequest(
		model=model_name,
		inputs=input_texts,
		dimensions=None,
		batch_size=32,  # Client batch_size hint, provider uses its own logic
		encoding_format=None,
		input_type=None,
		truncate=None,
	)

	# Mock Voyage API response for the batch
	# Voyage returns a list of embeddings directly in the 'data' field without 'object' or 'index' per item.
	# The VoyageAPIResponse model expects data: list[VoyageEmbeddingData]
	# VoyageEmbeddingData has object, embedding, index.
	# Let's align the mock with how _process_embeddings_response will receive it after model_validate
	mock_api_response_payload = {
		"object": "list",
		"data": [
			{"object": "embedding", "embedding": [0.1] * 10, "index": 0},
			{"object": "embedding", "embedding": [0.2] * 10, "index": 1},
		],
		"model": model_name,
		"usage": {"total_tokens": 4},
	}
	httpx_mock.add_response(
		url=f"{voyage_client.base_url}/embeddings",
		method="POST",
		json=mock_api_response_payload,
		status_code=200,
	)

	async with httpx.AsyncClient() as http_client:
		response = await voyage_client.aembed_batch(request_data, http_client)

	assert isinstance(response, BatchEmbeddingResponse)
	assert len(response.embeddings) == 2
	assert response.texts == input_texts
	assert response.model == model_name
	assert response.usage == {"total_tokens": 4}
	assert response.failed_indices is None or len(response.failed_indices) == 0

	# Check that one API call was made
	assert httpx_mock.get_requests() is not None
	all_requests = httpx_mock.get_requests()
	assert len(all_requests) == 1
	# Optionally, verify the content of the API request payload
	# import json
	# sent_payload = json.loads(all_requests[0].content)
	# assert sent_payload["input"] == input_texts
	# assert sent_payload["model"] == model_name


@pytest.mark.asyncio
async def test_voyage_client_aembed_batch_more_than_max_size(httpx_mock, mocker):
	"""Test VoyageClient.aembed_batch with input size > VOYAGE_INPUT_BATCH_SIZE, forcing multiple API calls."""
	# Patch the constant VOYAGE_INPUT_BATCH_SIZE within the provider module for this test
	mocker.patch.object(voyage_provider, "VOYAGE_API_MAX_BATCH_SIZE", 1)  # Force 1 text per API batch

	api_key = "test_voyage_aembed_batch_large"
	model_name = "voyage-large-2"
	provider_config = ProviderModel(
		provider="voyage", api_key=api_key, model=model_name, base_url=None, timeout=60, max_retries=3
	)
	voyage_client = VoyageClient(config=provider_config)

	input_texts = ["text one", "text two", "text three"]  # Will require 3 API calls
	request_data = BatchEmbeddingRequest(
		model=model_name,
		inputs=input_texts,
		dimensions=None,
		batch_size=32,  # Client-side batch_size hint, not used by provider's internal batching here
		encoding_format=None,
		input_type=None,
		truncate=None,
	)

	# Mock responses for each API call
	# Call 1 (for "text one")
	httpx_mock.add_response(
		url=f"{voyage_client.base_url}/embeddings",
		method="POST",
		status_code=200,
		json={
			"object": "list",
			"data": [{"object": "embedding", "embedding": [0.1] * 5, "index": 0}],
			"model": model_name,
			"usage": {"total_tokens": 1},
		},
	)
	# Call 2 (for "text two")
	httpx_mock.add_response(
		url=f"{voyage_client.base_url}/embeddings",
		method="POST",
		status_code=200,
		json={
			"object": "list",
			"data": [{"object": "embedding", "embedding": [0.2] * 5, "index": 0}],
			"model": model_name,
			"usage": {"total_tokens": 1},
		},
	)
	# Call 3 (for "text three")
	httpx_mock.add_response(
		url=f"{voyage_client.base_url}/embeddings",
		method="POST",
		status_code=200,
		json={
			"object": "list",
			"data": [{"object": "embedding", "embedding": [0.3] * 5, "index": 0}],
			"model": model_name,
			"usage": {"total_tokens": 1},
		},
	)

	async with httpx.AsyncClient() as http_client:
		response = await voyage_client.aembed_batch(request_data, http_client)

	assert isinstance(response, BatchEmbeddingResponse)
	assert len(response.embeddings) == 3
	assert response.texts == input_texts
	assert response.model == model_name
	assert response.usage == {"total_tokens": 3}
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
async def test_voyage_client_aembed_batch_partial_failure_multi_calls(httpx_mock, mocker):
	"""Test VoyageClient.aembed_batch with one internal call failing and others succeeding."""
	mocker.patch.object(voyage_provider, "VOYAGE_API_MAX_BATCH_SIZE", 1)  # Each text is a batch

	api_key = "test_voyage_aembed_batch_partial_fail"
	model_name = "voyage-2"
	provider_config = ProviderModel(
		provider="voyage", api_key=api_key, model=model_name, base_url=None, timeout=60, max_retries=3
	)
	voyage_client = VoyageClient(config=provider_config)

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
		url=f"{voyage_client.base_url}/embeddings",
		method="POST",
		status_code=200,
		json={
			"object": "list",
			"data": [{"object": "embedding", "embedding": [0.1] * 5, "index": 0}],
			"model": model_name,
			"usage": {"total_tokens": 1},
		},
	)
	# Call 2 (failure for "text fail 2")
	httpx_mock.add_response(
		url=f"{voyage_client.base_url}/embeddings",
		method="POST",
		status_code=400,  # Bad request
		json={"detail": "Invalid input for text fail 2"},  # Voyage error detail format
	)
	# Call 3 (success for "text success 3")
	httpx_mock.add_response(
		url=f"{voyage_client.base_url}/embeddings",
		method="POST",
		status_code=200,
		json={
			"object": "list",
			"data": [{"object": "embedding", "embedding": [0.3] * 5, "index": 0}],
			"model": model_name,
			"usage": {"total_tokens": 1},
		},
	)

	# The failing call (batch 2) won't reach model_validate in _process_embeddings_response
	# because raise_for_status() happens first in the provider's aembed_batch loop.
	# No need to mock VoyageAPIResponse.model_validate here explicitly unless non-HTTP errors are tested.

	async with httpx.AsyncClient() as http_client:
		response = await voyage_client.aembed_batch(request_data, http_client)

	assert isinstance(response, BatchEmbeddingResponse)
	assert len(response.embeddings) == 2  # Two successes
	assert response.failed_indices is not None
	# Cast to list[int] after the None check to satisfy type checker
	failed_list_for_check = cast("list[int]", response.failed_indices)
	assert len(failed_list_for_check) == 1
	assert 1 in failed_list_for_check  # Original index 1 ("text fail 2") failed

	assert response.model == model_name
	assert response.texts == input_texts  # Should contain all original texts
	assert response.usage == {"total_tokens": 2}  # Corrected, sum of 2 successful calls

	# Check successful embeddings (original indices 0 and 2)
	assert response.embeddings[0].values == [0.1] * 5
	assert response.embeddings[0].index == 0
	assert response.embeddings[1].values == [0.3] * 5
	assert response.embeddings[1].index == 2

	assert httpx_mock.get_requests() is not None
	assert len(httpx_mock.get_requests()) == 3  # All three API calls attempted


# Synchronous batch tests for VoyageClient


def test_voyage_client_embed_batch_less_than_max_size(httpx_mock):
	"""Test VoyageClient.embed_batch with input size < VOYAGE_INPUT_BATCH_SIZE."""
	api_key = "test_voyage_embed_batch_small"
	model_name = "voyage-2"
	provider_config = ProviderModel(
		provider="voyage", api_key=api_key, model=model_name, base_url=None, timeout=60, max_retries=3
	)
	voyage_client = VoyageClient(config=provider_config)
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
	mock_api_response_payload = {
		"object": "list",
		"data": [
			{"object": "embedding", "embedding": [0.5] * 10, "index": 0},
			{"object": "embedding", "embedding": [0.6] * 10, "index": 1},
		],
		"model": model_name,
		"usage": {"total_tokens": 5},
	}
	httpx_mock.add_response(
		url=f"{voyage_client.base_url}/embeddings", method="POST", json=mock_api_response_payload, status_code=200
	)

	with httpx.Client() as http_client:
		response = voyage_client.embed_batch(request_data, http_client)

	assert isinstance(response, BatchEmbeddingResponse)
	assert len(response.embeddings) == 2
	assert response.texts == input_texts
	assert response.model == model_name
	assert response.usage == {"total_tokens": 5}
	assert not response.failed_indices
	assert len(httpx_mock.get_requests()) == 1


def test_voyage_client_embed_batch_more_than_max_size(httpx_mock, mocker):
	"""Test VoyageClient.embed_batch with input size > VOYAGE_INPUT_BATCH_SIZE."""
	mocker.patch.object(voyage_provider, "VOYAGE_API_MAX_BATCH_SIZE", 1)
	api_key = "test_voyage_embed_batch_large"
	model_name = "voyage-large-2"
	provider_config = ProviderModel(
		provider="voyage", api_key=api_key, model=model_name, base_url=None, timeout=60, max_retries=3
	)
	voyage_client = VoyageClient(config=provider_config)
	input_texts = ["sync text one", "sync text two", "sync text three"]
	request_data = BatchEmbeddingRequest(
		model=model_name,
		inputs=input_texts,
		dimensions=None,
		batch_size=32,
		encoding_format=None,
		input_type=None,
		truncate=None,
	)

	httpx_mock.add_response(
		url=f"{voyage_client.base_url}/embeddings",
		method="POST",
		status_code=200,
		json={
			"object": "list",
			"data": [{"object": "embedding", "embedding": [0.1] * 5, "index": 0}],
			"model": model_name,
			"usage": {"total_tokens": 1},
		},
	)
	httpx_mock.add_response(
		url=f"{voyage_client.base_url}/embeddings",
		method="POST",
		status_code=200,
		json={
			"object": "list",
			"data": [{"object": "embedding", "embedding": [0.2] * 5, "index": 0}],
			"model": model_name,
			"usage": {"total_tokens": 1},
		},
	)
	httpx_mock.add_response(
		url=f"{voyage_client.base_url}/embeddings",
		method="POST",
		status_code=200,
		json={
			"object": "list",
			"data": [{"object": "embedding", "embedding": [0.3] * 5, "index": 0}],
			"model": model_name,
			"usage": {"total_tokens": 1},
		},
	)

	with httpx.Client() as http_client:
		response = voyage_client.embed_batch(request_data, http_client)

	assert isinstance(response, BatchEmbeddingResponse)
	assert len(response.embeddings) == 3
	assert response.texts == input_texts
	assert response.model == model_name
	assert response.usage == {"total_tokens": 3}
	assert not response.failed_indices
	assert response.embeddings[0].values == [0.1] * 5
	assert response.embeddings[0].index == 0
	assert response.embeddings[1].values == [0.2] * 5
	assert response.embeddings[1].index == 1
	assert response.embeddings[2].values == [0.3] * 5
	assert response.embeddings[2].index == 2
	assert len(httpx_mock.get_requests()) == 3


def test_voyage_client_embed_batch_partial_failure_multi_calls(httpx_mock, mocker):
	"""Test VoyageClient.embed_batch with partial failure."""
	mocker.patch.object(voyage_provider, "VOYAGE_API_MAX_BATCH_SIZE", 1)
	api_key = "test_voyage_embed_batch_partial_fail"
	model_name = "voyage-2"
	provider_config = ProviderModel(
		provider="voyage", api_key=api_key, model=model_name, base_url=None, timeout=60, max_retries=3
	)
	voyage_client = VoyageClient(config=provider_config)
	input_texts = ["sync success 1", "sync fail 2", "sync success 3"]
	request_data = BatchEmbeddingRequest(
		model=model_name,
		inputs=input_texts,
		dimensions=None,
		batch_size=32,
		encoding_format=None,
		input_type=None,
		truncate=None,
	)

	httpx_mock.add_response(
		url=f"{voyage_client.base_url}/embeddings",
		method="POST",
		status_code=200,
		json={
			"object": "list",
			"data": [{"object": "embedding", "embedding": [0.1] * 5, "index": 0}],
			"model": model_name,
			"usage": {"total_tokens": 1},
		},
	)
	httpx_mock.add_response(
		url=f"{voyage_client.base_url}/embeddings",
		method="POST",
		status_code=400,
		json={"detail": "Invalid for sync fail 2"},
	)
	httpx_mock.add_response(
		url=f"{voyage_client.base_url}/embeddings",
		method="POST",
		status_code=200,
		json={
			"object": "list",
			"data": [{"object": "embedding", "embedding": [0.3] * 5, "index": 0}],
			"model": model_name,
			"usage": {"total_tokens": 1},
		},
	)

	with httpx.Client() as http_client:
		response = voyage_client.embed_batch(request_data, http_client)

	assert isinstance(response, BatchEmbeddingResponse)
	assert len(response.embeddings) == 2
	assert response.failed_indices is not None
	# Cast to list[int] after the None check to satisfy type checker
	failed_list_for_check_2 = cast("list[int]", response.failed_indices)
	assert len(failed_list_for_check_2) == 1
	assert 1 in failed_list_for_check_2

	assert response.model == model_name
	assert response.texts == input_texts
	assert response.usage == {"total_tokens": 2}
	assert response.embeddings[0].values == [0.1] * 5
	assert response.embeddings[0].index == 0
	assert response.embeddings[1].values == [0.3] * 5
	assert response.embeddings[1].index == 2
	assert len(httpx_mock.get_requests()) == 3
