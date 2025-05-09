"""
Base models for the Embed-All library.

This module contains the base model classes used throughout the library.
"""

from pydantic import BaseModel as PydanticBaseModel
from pydantic import ConfigDict, Field


class BaseModel(PydanticBaseModel):
	"""Base model for all models in the Embed-All library."""

	model_config = ConfigDict(
		populate_by_name=True,
		extra="ignore",
		str_strip_whitespace=True,
	)


class ProviderModel(BaseModel):
	"""Base model for provider-specific configuration."""

	provider: str = Field(..., description="The provider name")
	api_key: str | None = Field(None, description="API key for the provider")
	base_url: str | None = Field(None, description="Base URL for API requests")
	model: str = Field(..., description="Model name/identifier for embeddings")
	timeout: int = Field(60, description="Request timeout in seconds")
	max_retries: int = Field(3, description="Maximum number of retries for failed requests")


class EmbeddingModel(BaseModel):
	"""Base model for embedding vectors."""

	values: list[float] = Field(..., description="The embedding vector values")
	dimensions: int = Field(..., description="Number of dimensions in the embedding")
	index: int | None = Field(None, description="Index of this embedding in a batch")
	text: str | None = Field(None, description="Original text that was embedded")
	model: str | None = Field(None, description="Model used to generate the embedding")

	def __len__(self) -> int:
		"""Return the number of dimensions in the embedding."""
		return len(self.values)

	def to_list(self) -> list[float]:
		"""Return the embedding as a list of floats."""
		return self.values
