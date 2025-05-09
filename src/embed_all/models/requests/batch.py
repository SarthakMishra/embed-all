"""
Batch embedding request models for the Embed-All library.

This module contains request models for batch embeddings across different providers.
"""

from typing import Literal

from pydantic import Field, field_validator

from embed_all.models.requests.base import BaseRequest


class BatchEmbeddingRequest(BaseRequest):
	"""Request model for batch embedding operations."""

	inputs: list[str] = Field(..., description="List of texts to embed in batch")
	dimensions: int | None = Field(None, description="Desired dimensionality of the embeddings")
	batch_size: int = Field(32, description="Number of texts to process in each batch")
	truncate: Literal["NONE", "START", "END"] | None = Field(
		"NONE", description="Truncation strategy if text exceeds model's context window"
	)
	encoding_format: Literal["float", "base64"] | None = Field("float", description="Output encoding format")

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
