"""
Request models for the Embed-All library.

This package contains all the request models used for different embedding providers.
"""

from embed_all.models.requests.base import BaseRequest
from embed_all.models.requests.batch import BatchEmbeddingRequest
from embed_all.models.requests.multimodal import MultimodalEmbeddingRequest
from embed_all.models.requests.text import TextEmbeddingRequest

__all__ = [
	"BaseRequest",
	"BatchEmbeddingRequest",
	"MultimodalEmbeddingRequest",
	"TextEmbeddingRequest",
]
