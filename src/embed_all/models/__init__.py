"""
Embed-All Models package.

This package contains all the model definitions for the Embed-All library.
"""

from embed_all.models.base import BaseModel, EmbeddingModel, ProviderModel
from embed_all.models.errors import (
	APIError,
	AuthenticationError,
	EmbedError,
	InvalidRequestError,
	RateLimitError,
)

# Import and re-export request models
from embed_all.models.requests.base import BaseRequest
from embed_all.models.requests.batch import BatchEmbeddingRequest
from embed_all.models.requests.multimodal import MultimodalEmbeddingRequest
from embed_all.models.requests.text import TextEmbeddingRequest

# Import and re-export response models
from embed_all.models.responses.base import BaseResponse
from embed_all.models.responses.batch import BatchEmbeddingResponse
from embed_all.models.responses.multimodal import MultimodalEmbeddingResponse
from embed_all.models.responses.text import TextEmbeddingResponse

__all__ = [
	"APIError",
	"AuthenticationError",
	"BaseModel",
	# Request models
	"BaseRequest",
	# Response models
	"BaseResponse",
	"BatchEmbeddingRequest",
	"BatchEmbeddingResponse",
	# Error models
	"EmbedError",
	# Base models
	"EmbeddingModel",
	"InvalidRequestError",
	"MultimodalEmbeddingRequest",
	"MultimodalEmbeddingResponse",
	"ProviderModel",
	"RateLimitError",
	"TextEmbeddingRequest",
	"TextEmbeddingResponse",
]
