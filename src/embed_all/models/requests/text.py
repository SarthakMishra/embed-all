"""
Text embedding request models for the Embed-All library.

This module contains request models for text embeddings across different providers.
"""

from typing import Self

from pydantic import Field, field_validator, model_validator

from embed_all.models.requests.base import BaseRequest


class TextEmbeddingRequest(BaseRequest):
	"""Request model for text embedding operations."""

	input: str | list[str] = Field(..., description="Text or list of texts to embed.")
	model: str = Field(..., description="Model to use for embedding.")
	truncate: bool | None = Field(
		None, description="Whether to truncate the input text if it exceeds the model's context length."
	)
	encoding_format: str | None = Field(None, description="The format of the embeddings.")
	dimensions: int | None = Field(None, description="The desired dimension of the output embeddings.")
	input_type: str | None = Field(
		None, description="Type of the input text (e.g., query, document). Provider-specific."
	)

	@model_validator(mode="after")
	def check_input_not_empty(self) -> Self:
		"""Validate the input text(s)."""
		if isinstance(self.input, str):
			if not self.input.strip():
				msg = "Input text cannot be empty"
				raise ValueError(msg)
		elif isinstance(self.input, list):
			if not self.input:
				msg = "Input list cannot be empty"
				raise ValueError(msg)
			for text_item in self.input:
				if not isinstance(text_item, str):
					msg = f"All inputs must be strings, got {type(text_item)}"
					raise TypeError(msg)
				if not text_item.strip():
					msg = "Input texts cannot be empty"
					raise ValueError(msg)
		else:
			msg = f"Input must be a string or list of strings, got {type(self.input)}"
			raise TypeError(msg)
		return self

	@field_validator("dimensions")
	@classmethod
	def validate_dimensions(cls, v: int | None) -> int | None:
		"""Validate the dimensions parameter."""
		if v is not None and v <= 0:
			msg = "Dimensions must be a positive integer"
			raise ValueError(msg)
		return v
