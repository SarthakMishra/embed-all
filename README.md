
# Embed-All
[![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)


> [!Caution]
> Embed-All is currently in active development. Use with caution and expect breaking changes.

## Overview

**Embed-All** is a light, type-safe Python wrapper around popular embedding model APIs. It provides a unified, minimal-dependency interface for generating text embeddings from multiple providers, with both synchronous and asynchronous clients.

## Features

- **Unified API** for multiple embedding providers (OpenAI, Voyage AI, and more coming soon)
- **Type-safe** request and response models (Pydantic)
- **Sync and async** client support
- **Batch and single** embedding support
- **Minimal dependencies**
- **Extensible**: add new providers easily

## Supported Providers

| Provider      | Provider Key | Default Model          |
|---------------|--------------|------------------------|
| OpenAI        | `openai`     | `text-embedding-3-small` |
| Voyage AI     | `voyage`     | `voyage-2`             |
| Ollama         | `ollama`     | `all-minilm`          |
| Cohere         | `cohere`     | `embed-v4.0`          |

## Installation

> **Requires Python 3.12+**

Install using [uv](https://github.com/astral-sh/uv) (recommended):

```bash
uv add git+https://github.com/SarthakMishra/embed-all.git
```

Or, for development:

```bash
git clone https://github.com/SarthakMishra/embed-all.git
cd embed-all

# Initialize a virtual env
uv venv
source .venv/bin/activate

# Sync all deps
uv sync --all-groups
```

## Quickstart

### Synchronous Usage

```python
from embed_all import embed

# Quick embedding (convenience function)
embeddings = embed(
    text=["hello world", "embed this!"],
    provider="openai",  # or "voyage"
    api_key="sk-...",
    model="text-embedding-3-small",  # optional, uses provider default if omitted
)
print(embeddings)  # List of lists of floats
```

Or, using the client directly:

```python
from embed_all import Client

with Client(provider="voyage", api_key="your-voyage-key") as client:
    response = client.embed("embed this text")
    print(response.to_list())  # List of lists of floats
```

### Asynchronous Usage

```python
import asyncio
from embed_all import AsyncClient

async def main():
    async with AsyncClient(provider="openai", api_key="sk-...") as client:
        response = await client.aembed(["async embedding", "another text"])
        print(response.to_list())

asyncio.run(main())
```

## API Reference

### Convenience Function

```python
embed(
    text: str | list[str],
    provider: str = "voyage",
    api_key: str | None = None,
    model: str | None = None,
    dimensions: int | None = None,
    **kwargs
) -> list[list[float]]
```

### Client Classes

- `Client` (sync)
- `AsyncClient` (async)

#### Methods

- `embed(text, model=None, dimensions=None, **kwargs)`
- `embed_batch(texts, model=None, dimensions=None, batch_size=32, **kwargs)`
- `aembed(...)` / `aembed_batch(...)` (async only)

#### Request Parameters

- `text` / `texts`: str or list of str
- `model`: model name (optional)
- `dimensions`: output embedding size (optional)
- `encoding_format`: "float" or "base64" (optional, provider-specific)
- `truncate`: bool (optional, provider-specific)
- `input_type`: str (optional, provider-specific)

#### Response

- `.to_list()`: returns all embeddings as a list of lists of floats
- `.embeddings`: list of `EmbeddingModel` (with `.values`, `.dimensions`, `.text`, etc.)

### Error Handling

All errors inherit from `EmbedError`. Common exceptions:

- `AuthenticationError`
- `RateLimitError`
- `InvalidRequestError`
- `APIError`

## Development

- **Lint:** `task lint` or `task lint:fix`
- **Format:** `task format`
- **Type-check:** `task lint:pyright`
- **Test:** `task test`
- **CI (all checks):** `task ci`

See `Taskfile.yml` for all available tasks.

## Contributing

1. Fork and clone the repo
2. Install dependencies with `uv pip install -e .`
3. Run `task ci` before submitting a PR
4. Follow the code style and type hints (see `.ruff` and `pyright` configs)

## License

[MIT](LICENSE) © 2024 Sarthak & Contributors