"""
Multimodal embedding response models for the Embed-All library.

This module contains response models for multimodal embeddings across different providers.
"""

from pydantic import Field

from embed_all.models.base import EmbeddingModel
from embed_all.models.responses.base import BaseResponse


class MultimodalEmbeddingResponse(BaseResponse):
	"""Response model for multimodal embeddings."""

	embeddings: list[EmbeddingModel] = Field(..., description="List of embeddings")
	dimensions: int = Field(..., description="Number of dimensions in each embedding")
	input_types: list[str] = Field(..., description="Types of inputs that were embedded (text, image, etc.)")

	def __len__(self) -> int:
		"""Return the number of embeddings."""
		return len(self.embeddings)

	def __getitem__(self, idx: int) -> EmbeddingModel:
		"""Get embedding by index."""
		return self.embeddings[idx]

	def to_list(self) -> list[list[float]]:
		"""Return all embeddings as a list of lists of floats."""
		return [embedding.values for embedding in self.embeddings]
