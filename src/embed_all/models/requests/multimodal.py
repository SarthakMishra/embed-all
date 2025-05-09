"""
Multimodal embedding request models for the Embed-All library.

This module contains request models for multimodal embeddings across different providers.
"""

from typing import Literal

from pydantic import Field, field_validator, model_validator

from embed_all.models.base import BaseModel
from embed_all.models.requests.base import BaseRequest


class ImageInput(BaseModel):
	"""Model for image input data."""

	url: str | None = Field(None, description="URL of the image to embed")
	base64: str | None = Field(None, description="Base64-encoded image data")
	path: str | None = Field(None, description="Local file path to the image")

	@model_validator(mode="after")
	def validate_image_source(self):
		"""Validate that exactly one image source is provided."""
		sources = [s for s in [self.url, self.base64, self.path] if s is not None]
		if len(sources) == 0:
			msg = "At least one of url, base64, or path must be provided"
			raise ValueError(msg)
		if len(sources) > 1:
			msg = "Only one of url, base64, or path should be provided"
			raise ValueError(msg)
		return self


class MultimodalInput(BaseModel):
	"""Model for multimodal input combining text and images."""

	text: str | None = Field(None, description="Text component of the multimodal input")
	image: ImageInput | None = Field(None, description="Image component of the multimodal input")

	@model_validator(mode="after")
	def validate_multimodal_input(self):
		"""Validate that at least one of text or image is provided."""
		if self.text is None and self.image is None:
			msg = "At least one of text or image must be provided"
			raise ValueError(msg)
		return self


class MultimodalEmbeddingRequest(BaseRequest):
	"""Request model for multimodal embeddings."""

	input: MultimodalInput | list[MultimodalInput] = Field(
		..., description="Multimodal input or list of inputs to embed"
	)
	dimensions: int | None = Field(None, description="Desired dimensionality of the embeddings")
	encoding_format: Literal["float", "base64"] | None = Field("float", description="Output encoding format")

	@field_validator("input")
	@classmethod
	def validate_input(cls, v: MultimodalInput | list[MultimodalInput]) -> MultimodalInput | list[MultimodalInput]:
		"""Validate the input(s)."""
		if isinstance(v, list) and not v:
			msg = "Input list cannot be empty"
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
