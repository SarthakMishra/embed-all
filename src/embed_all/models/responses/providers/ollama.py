"""Pydantic models for Ollama API responses."""

from pydantic import BaseModel


class OllamaEmbeddingResponse(BaseModel):
	"""Represents the response from the Ollama embedding API."""

	model: str
	embeddings: list[list[float]]
	total_duration: int | None = None
	load_duration: int | None = None
	prompt_eval_count: int | None = None
