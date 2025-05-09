"""
Embed-All Client package.

This package contains client implementations for different embedding providers.
"""

from embed_all.client.async_client import AsyncClient
from embed_all.client.sync_client import Client

__all__ = ["AsyncClient", "Client"]
