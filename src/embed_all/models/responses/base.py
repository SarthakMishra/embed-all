"""
Base response models for the Embed-All library.

This module contains base response models used for all embedding providers.
"""

from typing import Any

from pydantic import Field

from embed_all.models.base import BaseModel


class BaseResponse(BaseModel):
	"""Base class for all response models."""

	model: str = Field(default=..., description="The model used for embedding")
	provider: str = Field(default=..., description="The provider that generated the embeddings")
	usage: dict[str, Any] | None = Field(default=None, description="Usage information from the provider")
