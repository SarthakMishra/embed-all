"""Tests for src/embed_all/models/requests.py."""

import pytest
from pydantic import ValidationError

from embed_all.models import (
	BaseRequest,
	BatchEmbeddingRequest,
	TextEmbeddingRequest,
)


# Tests for BaseRequest (formerly BaseEmbeddingRequest)
def test_base_request_missing_model():
	"""Test BaseRequest raises ValidationError if 'model' is missing."""
	with pytest.raises(ValidationError) as excinfo:
		# If BaseRequest has only 'model' as required, this is correct. If more required, add dummy values for them except 'model'.
		BaseRequest()  # type: ignore[call-arg]
	assert "model" in str(excinfo.value).lower()


def test_base_request_with_model():
	"""Test BaseRequest successful instantiation with 'model'."""
	try:
		req = BaseRequest(model="test-model")
		assert req.model == "test-model"
	except ValidationError:
		pytest.fail("BaseRequest failed with required 'model' field provided.")


def test_base_request_empty_model_fails():
	"""Test BaseRequest validator fails for empty model string."""
	with pytest.raises(ValidationError, match="Model name cannot be empty"):
		BaseRequest(model="")
	with pytest.raises(ValidationError, match="Model name cannot be empty"):
		BaseRequest(model="   ")


# Tests for TextEmbeddingRequest
def test_text_embedding_request_missing_input():
	"""Test TextEmbeddingRequest raises ValidationError if 'input' is missing."""
	with pytest.raises(ValidationError) as excinfo:
		# Provide all required except 'input'
		TextEmbeddingRequest(model="test-model")  # type: ignore[call-arg]
	assert "input" in str(excinfo.value).lower()


def test_text_embedding_request_missing_model():
	"""Test TextEmbeddingRequest raises ValidationError if 'model' (from base) is missing."""
	with pytest.raises(ValidationError) as excinfo:
		# Provide all required except 'model'
		TextEmbeddingRequest(input="some text")  # type: ignore[call-arg]
	assert "model" in str(excinfo.value).lower()


def test_text_embedding_request_valid_single_input():
	"""Test TextEmbeddingRequest successful instantiation with single string input."""
	try:
		req = TextEmbeddingRequest(
			model="test-model", input="hello", dimensions=None, input_type=None, encoding_format=None, truncate=None
		)
		assert req.model == "test-model"
		assert req.input == "hello"
	except ValidationError:
		pytest.fail("TextEmbeddingRequest failed with valid single string input.")


def test_text_embedding_request_valid_list_input():
	"""Test TextEmbeddingRequest successful instantiation with list of strings input."""
	try:
		req = TextEmbeddingRequest(
			model="test-model",
			input=["hello", "world"],
			dimensions=None,
			input_type=None,
			encoding_format=None,
			truncate=None,
		)
		assert req.model == "test-model"
		assert req.input == ["hello", "world"]
	except ValidationError:
		pytest.fail("TextEmbeddingRequest failed with valid list of strings input.")


# Tests for BatchEmbeddingRequest
def test_batch_embedding_request_missing_inputs():
	"""Test BatchEmbeddingRequest raises ValidationError if 'inputs' is missing."""
	with pytest.raises(ValidationError) as excinfo:
		BatchEmbeddingRequest(model="test-model")
	assert "inputs" in str(excinfo.value).lower()


def test_batch_embedding_request_missing_model():
	"""Test BatchEmbeddingRequest raises ValidationError if 'model' is missing."""
	with pytest.raises(ValidationError) as excinfo:
		# Provide all required except 'model'
		BatchEmbeddingRequest(inputs=["batch text"])  # type: ignore[call-arg]
	assert "model" in str(excinfo.value).lower()


def test_batch_embedding_request_valid():
	"""Test BatchEmbeddingRequest successful instantiation."""
	try:
		req = BatchEmbeddingRequest(
			model="test-model",
			inputs=["text1", "text2"],
			dimensions=None,
			input_type=None,
			encoding_format=None,
			truncate=None,
			# batch_size removed, will use default 32
		)
		assert req.model == "test-model"
		assert req.inputs == ["text1", "text2"]
		assert req.batch_size == 32  # Check for default value
	except ValidationError:
		pytest.fail("BatchEmbeddingRequest failed with valid inputs.")


def test_batch_embedding_request_with_batch_size():
	"""Test BatchEmbeddingRequest successful instantiation with batch_size specified."""
	try:
		req = BatchEmbeddingRequest(
			model="test-model",
			inputs=["text1"],
			batch_size=16,
			dimensions=None,
			input_type=None,
			encoding_format=None,
			truncate=None,
		)
		assert req.model == "test-model"
		assert req.inputs == ["text1"]
		assert req.batch_size == 16
	except ValidationError:
		pytest.fail("BatchEmbeddingRequest failed with batch_size specified.")


# Additional validator tests for BatchEmbeddingRequest
def test_batch_embedding_request_inputs_validator_empty_list():
	"""Test BatchEmbeddingRequest 'inputs' validator fails for empty list."""
	with pytest.raises(ValidationError, match="Inputs list cannot be empty"):
		# For pyright: provide all other optional fields as None if not defaulted, or rely on defaults
		BatchEmbeddingRequest(
			model="test-model", inputs=[], dimensions=None, input_type=None, encoding_format=None, truncate=None
		)


def test_batch_embedding_request_inputs_validator_not_all_strings():
	"""Test BatchEmbeddingRequest 'inputs' validator fails if not all items are strings."""
	with pytest.raises(ValidationError, match="Input should be a valid string"):
		BatchEmbeddingRequest(
			model="test-model",
			inputs=["text1", 123],  # type: ignore[list-item]
			dimensions=None,
			input_type=None,
			encoding_format=None,
			truncate=None,
		)


def test_batch_embedding_request_inputs_validator_empty_string_in_list():
	"""Test BatchEmbeddingRequest 'inputs' validator fails if any string in list is empty."""
	with pytest.raises(ValidationError, match="Input texts cannot be empty"):
		BatchEmbeddingRequest(
			model="test-model",
			inputs=["text1", ""],
			dimensions=None,
			input_type=None,
			encoding_format=None,
			truncate=None,
		)
	with pytest.raises(ValidationError, match="Input texts cannot be empty"):
		BatchEmbeddingRequest(
			model="test-model",
			inputs=["text1", "   "],
			dimensions=None,
			input_type=None,
			encoding_format=None,
			truncate=None,
		)


def test_batch_embedding_request_batch_size_validator_non_positive():
	"""Test BatchEmbeddingRequest 'batch_size' validator fails for non-positive values."""
	with pytest.raises(ValidationError, match="Batch size must be a positive integer"):
		BatchEmbeddingRequest(
			model="test-model",
			inputs=["text"],
			batch_size=0,
			dimensions=None,
			input_type=None,
			encoding_format=None,
			truncate=None,
		)
	with pytest.raises(ValidationError, match="Batch size must be a positive integer"):
		BatchEmbeddingRequest(
			model="test-model",
			inputs=["text"],
			batch_size=-1,
			dimensions=None,
			input_type=None,
			encoding_format=None,
			truncate=None,
		)


def test_batch_embedding_request_dimensions_validator_non_positive():
	"""Test BatchEmbeddingRequest 'dimensions' validator fails for non-positive values."""
	with pytest.raises(ValidationError, match="Dimensions must be a positive integer"):
		BatchEmbeddingRequest(
			model="test-model", inputs=["text"], dimensions=0, input_type=None, encoding_format=None, truncate=None
		)
	with pytest.raises(ValidationError, match="Dimensions must be a positive integer"):
		BatchEmbeddingRequest(
			model="test-model", inputs=["text"], dimensions=-5, input_type=None, encoding_format=None, truncate=None
		)


def test_batch_embedding_request_dimensions_validator_positive_ok():
	"""Test BatchEmbeddingRequest 'dimensions' validator passes for positive value."""
	try:
		# If pyright flags batch_size here, it's not respecting Pydantic default
		BatchEmbeddingRequest(
			model="test-model", inputs=["text"], dimensions=128, input_type=None, encoding_format=None, truncate=None
		)
	except ValidationError as e:
		pytest.fail(f"BatchEmbeddingRequest failed for valid positive dimensions: {e}")


# Future tests could include validation for Literal fields if specific values are required
# or if there are custom validators (@validator decorator) in the models.
# For now, testing required fields is a good start.
