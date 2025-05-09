"""
Error classes for the Embed-All library.

This module contains exception classes used throughout the library for error handling.
"""

from typing import Any


class EmbedError(Exception):
	"""Base exception class for all Embed-All errors."""

	def __init__(
		self,
		message: str,
		*,
		provider: str | None = None,
		status_code: int | None = None,
		response: dict[str, Any] | None = None,
	) -> None:
		self.message = message
		self.provider = provider
		self.status_code = status_code
		self.response = response
		super().__init__(self.message)

	def __str__(self) -> str:
		msg = self.message
		if self.provider:
			msg = f"[{self.provider}] {msg}"
		if self.status_code:
			msg = f"{msg} (Status: {self.status_code})"
		return msg


class AuthenticationError(EmbedError):
	"""Raised when there's an issue with authentication credentials."""



class RateLimitError(EmbedError):
	"""Raised when API rate limits are exceeded."""

	def __init__(
		self,
		message: str,
		*,
		provider: str | None = None,
		status_code: int | None = None,
		response: dict[str, Any] | None = None,
		retry_after: int | None = None,
	) -> None:
		super().__init__(message, provider=provider, status_code=status_code, response=response)
		self.retry_after = retry_after


class InvalidRequestError(EmbedError):
	"""Raised when the request is invalid."""



class APIError(EmbedError):
	"""Raised when the API returns an error."""

