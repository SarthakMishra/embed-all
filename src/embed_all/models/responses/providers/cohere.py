"""Pydantic models for Cohere API responses."""

from typing import Any

from pydantic import BaseModel


class CohereMetaApiVersion(BaseModel):
	"""Represents the API version information in the Cohere API response."""

	version: str
	is_experimental: bool | None = None


class CohereMeta(BaseModel):
	"""Represents the metadata in the Cohere API response."""

	api_version: CohereMetaApiVersion
	warnings: list[str] | None = None
	billed_units: dict[str, int] | None = None


class CohereAPIResponse(BaseModel):
	"""Represents the overall structure of a successful Cohere embeddings API response."""

	id: str
	texts: list[str] | None = None
	images: list[dict[str, Any]] | None = None
	embeddings: dict[str, list[list[float]]]
	meta: CohereMeta
