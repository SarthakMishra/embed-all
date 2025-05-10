"""
Provider client factory for Embed-All.

This module contains functions to get provider-specific client implementations.
"""

from embed_all.client.providers.base import BaseProviderClient
from embed_all.client.providers.open_ai import OpenAIClient
from embed_all.client.providers.voyage import VoyageClient
from embed_all.models.errors import InvalidRequestError

# Registry of provider clients
PROVIDER_CLIENTS: dict[str, type[BaseProviderClient]] = {
	"voyage": VoyageClient,
	"openai": OpenAIClient,
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
