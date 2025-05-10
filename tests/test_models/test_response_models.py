"""Tests for src/embed_all/models/responses/*."""

import pytest
from pydantic import ValidationError

from embed_all.models import (
	EmbeddingModel,  # Needed for TextEmbeddingResponse and BatchEmbeddingResponse
)
from embed_all.models.responses.base import BaseResponse
from embed_all.models.responses.batch import BatchEmbeddingResponse
from embed_all.models.responses.text import TextEmbeddingResponse


# Tests for BaseResponse
def test_base_response_instantiation():
	"""Test BaseResponse successful instantiation and missing required fields."""
	# Valid instantiation
	try:
		response = BaseResponse(model="model-x", provider="provider-y", usage={"tokens": 10})
		assert response.model == "model-x"
		assert response.provider == "provider-y"
		assert response.usage == {"tokens": 10}
	except ValidationError as e:
		pytest.fail(f"BaseResponse failed with valid data: {e}")

	# Missing model
	with pytest.raises(ValidationError, match="model"):
		BaseResponse(provider="provider-y")

	# Missing provider
	with pytest.raises(ValidationError, match="provider"):
		BaseResponse(model="model-x")

	# Usage is optional
	try:
		response = BaseResponse(model="model-x", provider="provider-y")
		assert response.usage is None
	except ValidationError as e:
		pytest.fail(f"BaseResponse failed when usage is None: {e}")


# Tests for TextEmbeddingResponse properties
def test_text_embedding_response_instantiation_and_methods():
	"""Test TextEmbeddingResponse instantiation, required fields, and methods."""
	emb1_values = [0.1, 0.2]
	emb2_values = [0.3, 0.4]
	# EmbeddingModel: values, dimensions are required. index, text, model are optional.
	emb1 = EmbeddingModel(values=emb1_values, dimensions=2, index=0, text="text1", model="model-x")
	emb2 = EmbeddingModel(values=emb2_values, dimensions=2, index=1, text="text2", model="model-x")
	embeddings_list = [emb1, emb2]
	texts_list = ["text1", "text2"]

	# Valid instantiation
	try:
		response = TextEmbeddingResponse(
			model="model-x",
			provider="provider-y",
			embeddings=embeddings_list,
			dimensions=2,  # This should ideally be derived or validated against embeddings
			texts=texts_list,
			usage={"tokens": 20},
		)
		assert response.model == "model-x"
		assert response.provider == "provider-y"
		assert response.embeddings == embeddings_list
		assert response.dimensions == 2
		assert response.texts == texts_list
		assert response.usage == {"tokens": 20}

		# Test methods
		assert len(response) == 2
		assert response[0] == emb1
		assert response[1] == emb2
		assert response.to_list() == [emb1_values, emb2_values]

	except ValidationError as e:
		pytest.fail(f"TextEmbeddingResponse failed with valid data: {e}")

	# Test required fields (from TextEmbeddingResponse itself)
	base_args_for_text_resp = {
		"model": "model-x",
		"provider": "provider-y",
		"usage": None,
	}  # Add usage=None for type safety
	with pytest.raises(ValidationError, match="embeddings"):
		TextEmbeddingResponse(**base_args_for_text_resp, dimensions=2, texts=texts_list)

	with pytest.raises(ValidationError, match="dimensions"):
		TextEmbeddingResponse(**base_args_for_text_resp, embeddings=embeddings_list, texts=texts_list)

	with pytest.raises(ValidationError, match="texts"):
		TextEmbeddingResponse(**base_args_for_text_resp, embeddings=embeddings_list, dimensions=2)


# Tests for BatchEmbeddingResponse properties (e.g., success_rate)
def test_batch_embedding_response_instantiation_and_properties():
	"""Test BatchEmbeddingResponse instantiation, required fields, and properties."""
	emb1_values = [0.1, 0.2]
	emb2_values = [0.3, 0.4]
	emb1 = EmbeddingModel(values=emb1_values, dimensions=2, index=0, text="text1", model="model-x")
	emb2 = EmbeddingModel(values=emb2_values, dimensions=2, index=1, text="text2", model="model-x")
	embeddings_list_all_success = [emb1, emb2]
	embeddings_list_one_fail = [emb1]
	texts_list_all_success = ["text1", "text2"]
	texts_list_one_fail_original = ["text1", "text2-failed", "text3-also-failed-or-not-processed"]

	# Valid instantiation - all successful
	try:
		response_all_success = BatchEmbeddingResponse(
			model="model-x",
			provider="provider-y",
			embeddings=embeddings_list_all_success,
			dimensions=2,
			texts=texts_list_all_success,
			batch_count=1,
			failed_indices=None,
			usage={"tokens": 30},
		)
		assert response_all_success.model == "model-x"
		assert len(response_all_success.embeddings) == 2
		assert response_all_success.dimensions == 2
		assert response_all_success.texts == texts_list_all_success
		assert response_all_success.batch_count == 1
		assert response_all_success.failed_indices is None
		assert response_all_success.usage == {"tokens": 30}
		assert response_all_success.success_rate == 1.0
		assert len(response_all_success) == 2
		assert response_all_success[0] == emb1
		assert response_all_success.to_list() == [emb1_values, emb2_values]
	except ValidationError as e:
		pytest.fail(f"BatchEmbeddingResponse (all success) failed with valid data: {e}")

	# Valid instantiation - one failure
	try:
		response_one_fail = BatchEmbeddingResponse(
			model="model-x",
			provider="provider-y",
			embeddings=embeddings_list_one_fail,
			dimensions=2,
			texts=texts_list_one_fail_original,
			batch_count=2,
			failed_indices=[1, 2],
			usage={"tokens": 15},
		)
		assert len(response_one_fail.embeddings) == 1
		assert response_one_fail.texts == texts_list_one_fail_original
		assert response_one_fail.failed_indices == [1, 2]
		assert response_one_fail.success_rate == pytest.approx(1 / 3)
		assert len(response_one_fail) == 1
		assert response_one_fail[0] == emb1
		assert response_one_fail.to_list() == [emb1_values]
	except ValidationError as e:
		pytest.fail(f"BatchEmbeddingResponse (one fail) failed with valid data: {e}")

	# Test required fields
	base_args_for_batch = {
		"model": "model-x",
		"provider": "provider-y",
		"usage": None,
		"failed_indices": None,
	}  # Add all optional fields from BaseResponse and Batch itself
	with pytest.raises(ValidationError, match="embeddings"):
		BatchEmbeddingResponse(**base_args_for_batch, dimensions=2, texts=texts_list_all_success, batch_count=1)
	with pytest.raises(ValidationError, match="dimensions"):
		BatchEmbeddingResponse(
			**base_args_for_batch, embeddings=embeddings_list_all_success, texts=texts_list_all_success, batch_count=1
		)
	with pytest.raises(ValidationError, match="texts"):
		BatchEmbeddingResponse(
			**base_args_for_batch, embeddings=embeddings_list_all_success, dimensions=2, batch_count=1
		)
	with pytest.raises(ValidationError, match="batch_count"):
		BatchEmbeddingResponse(
			**base_args_for_batch, embeddings=embeddings_list_all_success, dimensions=2, texts=texts_list_all_success
		)
