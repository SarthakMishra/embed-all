"""Pydantic models for OpenAI API responses."""

from pydantic import BaseModel


class OpenAIEmbeddingData(BaseModel):
	"""Represents a single embedding object in the OpenAI API response."""

	object: str
	embedding: list[float]
	index: int


class OpenAIUsage(BaseModel):
	"""Represents the usage statistics in the OpenAI API response."""

	prompt_tokens: int
	total_tokens: int


class OpenAIAPIResponse(BaseModel):
	"""Represents the overall structure of a successful OpenAI embeddings API response."""

	object: str
	data: list[OpenAIEmbeddingData]
	model: str
	usage: OpenAIUsage
