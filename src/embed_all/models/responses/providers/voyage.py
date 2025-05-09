"""Pydantic models for Voyage AI API responses."""

from pydantic import BaseModel


class VoyageEmbeddingData(BaseModel):
	"""Represents a single embedding object in the Voyage AI API response."""

	object: str
	embedding: list[float | int]
	index: int


class VoyageUsage(BaseModel):
	"""Represents the usage statistics in the Voyage AI API response."""

	total_tokens: int


class VoyageAPIResponse(BaseModel):
	"""Represents the overall structure of a successful Voyage AI embeddings API response."""

	object: str
	data: list[VoyageEmbeddingData]
	model: str
	usage: VoyageUsage
