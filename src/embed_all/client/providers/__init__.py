"""
Provider client factory for Embed-All.

This module contains functions to get provider-specific client implementations.
"""

from embed_all.models.errors import InvalidRequestError

from .base import BaseProviderClient
from .cohere import CohereClient
from .ollama import OllamaClient
from .open_ai import OpenAIClient
from .voyage import VoyageClient

# Registry of provider clients
PROVIDER_CLIENTS = {
	"openai": OpenAIClient,
	"voyage": VoyageClient,
	"cohere": CohereClient,
	"ollama": OllamaClient,
}


def get_provider_client(provider: str) -> type[BaseProviderClient]:
	"""Get the provider-specific client class.

	Args:
		provider: The provider name

	Returns:
		The provider client class

	Raises:
		InvalidRequestError: If the provider is not supported
	"""
	provider_lower = provider.lower()
	if provider_lower not in PROVIDER_CLIENTS:
		supported = ", ".join(PROVIDER_CLIENTS.keys())
		msg = f"Provider '{provider}' is not supported. Supported providers: {supported}"
		raise InvalidRequestError(msg, provider=provider)
	return PROVIDER_CLIENTS[provider_lower]


__all__ = ["BaseProviderClient", "CohereClient", "OllamaClient", "OpenAIClient", "VoyageClient", "get_provider_client"]
