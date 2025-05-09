"""
Batch embedding request models for the Embed-All library.

This module contains request models for batch embeddings across different providers.
"""

from pydantic import Field, field_validator

from embed_all.models.requests.base import BaseRequest


class BatchEmbeddingRequest(BaseRequest):
	"""Request model for batch embedding operations."""

	inputs: list[str] = Field(..., description="List of texts to embed in batch")
	model: str
	dimensions: int | None = Field(None, description="Desired dimensionality of the embeddings")
	batch_size: int = Field(32, description="Number of texts to process in each batch")
	truncate: bool | None = Field(None, description="Whether to truncate the input text. Provider-specific handling.")
	encoding_format: str | None = Field(None, description="Output encoding format. Provider-specific.")
	input_type: str | None = Field(
		None, description="Type of the input text (e.g., query, document). Provider-specific."
	)

	@field_validator("inputs")
	@classmethod
	def validate_inputs(cls, v: list[str]) -> list[str]:
		"""Validate the input texts."""
		if not v:
			msg = "Inputs list cannot be empty"
			raise ValueError(msg)
		for text in v:
			if not isinstance(text, str):
				msg = f"All inputs must be strings, got {type(text)}"
				raise TypeError(msg)
			if text.strip() == "":
				msg = "Input texts cannot be empty"
				raise ValueError(msg)
		return v

	@field_validator("batch_size")
	@classmethod
	def validate_batch_size(cls, v: int) -> int:
		"""Validate the batch size."""
		if v <= 0:
			msg = "Batch size must be a positive integer"
			raise ValueError(msg)
		return v

	@field_validator("dimensions")
	@classmethod
	def validate_dimensions(cls, v: int | None) -> int | None:
		"""Validate the dimensions parameter."""
		if v is not None and v <= 0:
			msg = "Dimensions must be a positive integer"
			raise ValueError(msg)
		return v
