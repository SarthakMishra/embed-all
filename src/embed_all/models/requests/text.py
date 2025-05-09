"""
Text embedding request models for the Embed-All library.

This module contains request models for text embeddings across different providers.
"""

from typing import Literal

from pydantic import Field, field_validator

from embed_all.models.requests.base import BaseRequest


class TextEmbeddingRequest(BaseRequest):
	"""Request model for text embeddings."""

	input: str | list[str] = Field(..., description="Text or list of texts to embed")
	dimensions: int | None = Field(None, description="Desired dimensionality of the embeddings")
	truncate: Literal["NONE", "START", "END"] | None = Field(
		"NONE", description="Truncation strategy if text exceeds model's context window"
	)
	encoding_format: Literal["float", "base64"] | None = Field("float", description="Output encoding format")

	@field_validator("input")
	@classmethod
	def validate_input(cls, v: str | list[str]) -> str | list[str]:
		"""Validate the input text(s)."""
		if isinstance(v, str):
			if v.strip() == "":
				msg = "Input text cannot be empty"
				raise ValueError(msg)
			return v
		if isinstance(v, list):
			if not v:
				msg = "Input list cannot be empty"
				raise ValueError(msg)
			for text in v:
				if not isinstance(text, str):
					msg = f"All inputs must be strings, got {type(text)}"
					raise TypeError(msg)
				if text.strip() == "":
					msg = "Input texts cannot be empty"
					raise ValueError(msg)
			return v
		msg = f"Input must be a string or list of strings, got {type(v)}"
		raise TypeError(msg)

	@field_validator("dimensions")
	@classmethod
	def validate_dimensions(cls, v: int | None) -> int | None:
		"""Validate the dimensions parameter."""
		if v is not None and v <= 0:
			msg = "Dimensions must be a positive integer"
			raise ValueError(msg)
		return v
