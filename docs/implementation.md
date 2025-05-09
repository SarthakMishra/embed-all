# Embed-All Implementation Plan

## Overview

Embed-All is a light, type-safe wrapper around various embedding model APIs from different providers. It aims to be similar to LangChain but focused exclusively on embeddings with minimal dependencies. The library will use Pydantic for schema validation and HTTPX for API requests.

## Core Components

### 1. Client Architecture

```
src/embed_all/client/
├── __init__.py                  # Exports client classes
├── base_client.py               # Abstract base client with common functionality
├── sync_client.py               # Synchronous client implementation
├── async_client.py              # Asynchronous client implementation
└── providers/                   # Provider-specific implementations
    ├── __init__.py
    ├── voyage.py                # Voyage AI client
    ├── openai.py                # OpenAI client
    ├── cohere.py                # Cohere client
    ├── anthropic.py             # Anthropic client
    ├── huggingface.py           # HuggingFace client
    ├── ollama.py                # Ollama client (local deployment)
    └── sentence_transformers.py # Optional SentenceTransformers client
```

### 2. Data Models

```
src/embed_all/models/
├── __init__.py                # Exports all models
├── base.py                    # Base classes and interfaces
├── requests/                  # Request objects
│   ├── __init__.py
│   ├── base.py                # Base request class
│   ├── text.py                # Text embedding requests
│   ├── multimodal.py          # Multimodal embedding requests
│   └── batch.py               # Batch embedding requests
├── responses/                 # Response objects
│   ├── __init__.py
│   ├── base.py                # Base response class
│   ├── text.py                # Text embedding responses
│   ├── multimodal.py          # Multimodal embedding responses
│   └── batch.py               # Batch embedding responses
└── errors.py                  # Error classes
```

### 3. Configuration

```
src/embed_all/config/
├── __init__.py                # Exports configuration classes
├── settings.py                # Global settings
├── credentials.py             # API key management
└── provider_config.py         # Provider-specific configurations
```

### 4. Utilities

```
src/embed_all/utils/
├── __init__.py                # Exports utility functions
├── embedding_utils.py         # Embedding operations (similarity, search, etc.)
├── validation.py              # Input validation helpers
├── batching.py                # Batch processing utilities
├── http.py                    # HTTP request helpers
└── logger.py                  # Logging utilities
```

### 5. Schema Generation Scripts

```
scripts/
├── generate_models.py         # Main script to generate models from OpenAPI schemas
├── schema_parsers/            # Parsers for different schema formats
│   ├── __init__.py
│   ├── openapi.py             # OpenAPI schema parser
│   └── json_schema.py         # JSON schema parser
└── templates/                 # Jinja2 templates for code generation
    ├── __init__.py
    ├── model.py.jinja         # Template for model classes
    ├── client.py.jinja        # Template for client classes
    └── enum.py.jinja          # Template for enum classes
```

## Implementation Phases

### Phase 1: Core Infrastructure

1. **Define base interfaces and abstract classes**
   - Create abstract base client class
   - Define base request/response models
   - Implement error handling framework
   - Set up configuration management

2. **Implement HTTP layer**
   - Build HTTPX client wrapper with retry logic
   - Implement synchronous and asynchronous request handling
   - Add request/response logging

### Phase 2: First Provider Implementation (Voyage AI)

1. **Implement Voyage AI provider**
   - Create Voyage-specific request/response models
   - Implement embedding API endpoints
   - Add authentication handling
   - Write comprehensive tests

2. **Build utility functions**
   - Implement embedding similarity calculation
   - Add vector search capabilities
   - Create batching utilities

### Phase 3: Schema Parser & Generator

1. **Build OpenAPI schema parser**
   - Implement parser for OpenAPI specification
   - Create model generator from schema definitions
   - Add type conversion logic

2. **Create code generation pipeline**
   - Develop Jinja2 templates for code generation
   - Implement script to generate models from schemas
   - Add validation for generated code

### Phase 4: Additional Providers

1. **Add support for major embedding providers**
   - OpenAI
   - Cohere
   - Anthropic
   - HuggingFace
   - Google (VertexAI)

2. **Implement local model support**
   - Ollama integration
   - SentenceTransformers support (optional dependency)

### Phase 5: Testing and Documentation

1. **Comprehensive testing**
   - Unit tests for all components
   - Integration tests for each provider
   - Performance benchmarks

2. **Documentation**
   - API reference
   - Usage examples
   - Provider-specific guides

## API Design

### Client Usage Example

```python
# Synchronous client
from embed_all import Client

client = Client(
    provider="voyage",
    api_key="your-api-key",
    model="voyage-large-2"
)

# Get embeddings
embeddings = client.embed(
    texts=["This is a test", "Another test text"],
    dimensions=1024
)

# Calculate similarity
from embed_all.utils import cosine_similarity

similarity = cosine_similarity(embeddings[0], embeddings[1])
print(f"Similarity: {similarity}")

# Asynchronous client
from embed_all import AsyncClient
import asyncio

async def embed_texts():
    async_client = AsyncClient(
        provider="openai",
        api_key="your-api-key",
        model="text-embedding-3-large"
    )
    
    embeddings = await async_client.embed(
        texts=["This is a test", "Another test text"],
        dimensions=1536
    )
    
    return embeddings

embeddings = asyncio.run(embed_texts())
```

### Multi-Provider Support

```python
# Using multiple providers
from embed_all import Client

voyage_client = Client(
    provider="voyage",
    api_key="voyage-api-key",
    model="voyage-large-2"
)

openai_client = Client(
    provider="openai",
    api_key="openai-api-key",
    model="text-embedding-3-large"
)

# Compare embeddings from different providers
voyage_embedding = voyage_client.embed("This is a test text")
openai_embedding = openai_client.embed("This is a test text")

# Local model support
from embed_all import Client

ollama_client = Client(
    provider="ollama",
    base_url="http://localhost:11434",
    model="nomic-embed-text"
)

local_embedding = ollama_client.embed("This is a test text")
```

## Code Generation from OpenAPI Schemas

The schema generation process will:

1. Parse OpenAPI schemas from provider documentation
2. Generate Pydantic models for request/response objects
3. Create type-safe client methods for each API endpoint

Example script usage:

```bash
python -m scripts.generate_models --provider voyage --schema-path /path/to/schema.yml --output-dir src/embed_all/models/providers/voyage
```

## Dependencies

Core dependencies:
- `httpx`: HTTP client
- `pydantic`: Data validation and settings management

Optional dependencies:
- `ollama`: For local Ollama model support
- `sentence-transformers`: For local transformer models
- `jinja2`: For schema code generation (dev dependency)
- `pyyaml`: For OpenAPI schema parsing (dev dependency)

## Testing Strategy

1. **Unit Tests**
   - Test individual components in isolation
   - Mock external API calls
   - Validate model serialization/deserialization

2. **Integration Tests**
   - Test provider clients against real APIs
   - Validate authentication flows
   - Test error handling

3. **End-to-End Tests**
   - Test complete embedding workflows
   - Validate utility functions

## Documentation

1. **API Reference**
   - Document all public classes and methods
   - Include type information and examples

2. **Usage Guides**
   - Basic getting started guide
   - Provider-specific configuration
   - Advanced usage examples

3. **Contributing Guide**
   - How to add new providers
   - Testing requirements
   - Development setup

## Next Steps

1. Implement core infrastructure (base classes, interfaces)
2. Build Voyage AI provider integration
3. Develop schema parser and code generator
4. Add additional providers
5. Create comprehensive testing and documentation 