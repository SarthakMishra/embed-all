"""
Base request models for the Embed-All library.

This module contains base request models used for all embedding providers.
"""

from pydantic import Field, field_validator

from embed_all.models.base import BaseModel


class BaseRequest(BaseModel):
	"""Base class for all request models."""

	model: str = Field(..., description="The model to use for embedding")

	@field_validator("model")
	@classmethod
	def validate_model(cls, v: str) -> str:
		"""Validate the model name."""
		if not v or not isinstance(v, str) or v.strip() == "":
			msg = "Model name cannot be empty"
			raise ValueError(msg)
		return v
