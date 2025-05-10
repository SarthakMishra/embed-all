"""Tests for src/embed_all/client/providers/__init__.py."""

import pytest

from embed_all.client.providers import get_provider_client
from embed_all.client.providers.open_ai import OpenAIClient

# Import other provider clients as they are added and tested, e.g.:
# from embed_all.client.providers.voyage import VoyageClient
from embed_all.models import ProviderModel
from embed_all.models.errors import InvalidRequestError


@pytest.mark.parametrize("provider_name_case_variant", ["openai", "OpenAI", "OPENAI", "oPeNaI"])
def test_get_known_provider_case_insensitivity(provider_name_case_variant):
	"""Test get_provider_client returns correct client for known provider, case-insensitively."""
	# Minimal config needed for instantiation for this test
	_ = ProviderModel(
		provider=provider_name_case_variant,
		api_key="test_key",
		base_url=None,
		model=None,
		timeout=60,  # Explicitly added
		max_retries=3,  # Explicitly added
	)
	provider_client_class = get_provider_client(provider_name_case_variant)
	assert provider_client_class is OpenAIClient

	# Optionally, instantiate to ensure it doesn't raise an unexpected error, though not strictly testing get_provider_client here
	# client_instance = provider_client_class(config)
	# assert isinstance(client_instance, OpenAIClient)


def test_get_unknown_provider_raises_error():
	"""Test get_provider_client raises InvalidRequestError for an unknown provider."""
	unknown_provider_name = "non_existent_provider"
	with pytest.raises(InvalidRequestError) as excinfo:
		get_provider_client(unknown_provider_name)

	assert f"Provider '{unknown_provider_name}' is not supported" in str(excinfo.value)
	assert excinfo.value.provider == unknown_provider_name
