"""
Batch embedding response models for the Embed-All library.

This module contains response models for batch embeddings across different providers.
"""

from pydantic import Field

from embed_all.models.base import EmbeddingModel
from embed_all.models.responses.base import BaseResponse


class BatchEmbeddingResponse(BaseResponse):
	"""Response model for batch embedding operations."""

	embeddings: list[EmbeddingModel] = Field(..., description="List of embeddings")
	dimensions: int = Field(..., description="Number of dimensions in each embedding")
	texts: list[str] = Field(..., description="Original texts that were embedded")
	batch_count: int = Field(..., description="Number of batches processed")
	failed_indices: list[int] | None = Field(None, description="Indices of texts that failed to embed")

	def __len__(self) -> int:
		"""Return the number of embeddings."""
		return len(self.embeddings)

	def __getitem__(self, idx: int) -> EmbeddingModel:
		"""Get embedding by index."""
		return self.embeddings[idx]

	def to_list(self) -> list[list[float]]:
		"""Return all embeddings as a list of lists of floats."""
		return [embedding.values for embedding in self.embeddings]

	@property
	def success_rate(self) -> float:
		"""Return the percentage of successful embeddings."""
		if not self.failed_indices:
			return 1.0
		return (len(self.texts) - len(self.failed_indices)) / len(self.texts)
