# `https://github.com/voyage-ai/voyageai-python` reference

## Directory Structure
```
voyageai
├── api_resources/
│   ├── __init__.py
│   ├── api_requestor.py
│   ├── api_resource.py
│   ├── embedding.py
│   ├── multimodal_embedding.py
│   ├── reranking.py
│   └── response.py
├── object/
│   ├── __init__.py
│   ├── embeddings.py
│   ├── multimodal_embeddings.py
│   └── reranking.py
├── __init__.py
├── _base.py
├── client.py
├── client_async.py
├── embeddings_utils.py
├── error.py
├── py.typed
├── util.py
└── version.py
```

## Scanned Files
- [`https://github.com/voyage-ai/voyageai-python` reference](#httpsgithubcomvoyage-aivoyageai-python-reference)
	- [Directory Structure](#directory-structure)
	- [Scanned Files](#scanned-files)
	- [Code Documentation](#code-documentation)
		- [1. /__init__.py](#1-initpy)
		- [2. /\_base.py](#2-_basepy)
		- [3. /api\_resources/__init__.py](#3-api_resourcesinitpy)
		- [4. /api\_resources/api\_requestor.py](#4-api_resourcesapi_requestorpy)
		- [5. /api\_resources/api\_resource.py](#5-api_resourcesapi_resourcepy)
		- [6. /api\_resources/embedding.py](#6-api_resourcesembeddingpy)
		- [7. /api\_resources/multimodal\_embedding.py](#7-api_resourcesmultimodal_embeddingpy)
		- [8. /api\_resources/reranking.py](#8-api_resourcesrerankingpy)
		- [9. /api\_resources/response.py](#9-api_resourcesresponsepy)
		- [10. /client.py](#10-clientpy)
		- [11. /client\_async.py](#11-client_asyncpy)
		- [12. /embeddings\_utils.py](#12-embeddings_utilspy)
		- [13. /error.py](#13-errorpy)
		- [14. /object/__init__.py](#14-objectinitpy)
		- [15. /object/embeddings.py](#15-objectembeddingspy)
		- [16. /object/multimodal\_embeddings.py](#16-objectmultimodal_embeddingspy)
		- [17. /object/reranking.py](#17-objectrerankingpy)
		- [18. /util.py](#18-utilpy)
		- [19. /version.py](#19-versionpy)

## Code Documentation

### 1. /__init__.py
> *# Voyage AI Python bindings.*
> *#*
> *# Originally forked from the MIT-licensed OpenAI Python bindings.*
- **Import**: `os`
- **Import**: `sys`
- **Import**: `TYPE_CHECKING`
- **Import**: `ContextVar`
- **Constant**: `VOYAGE_EMBED_BATCH_SIZE`
  - **Constant**: `VOYAGE_EMBED_BATCH_SIZE`
- **Constant**: `VOYAGE_EMBED_DEFAULT_MODEL`
  - **Constant**: `VOYAGE_EMBED_DEFAULT_MODEL`
- **Import**: `Embedding`
- **Import**: `VERSION`
- **Import**: `Client`
- **Import**: `AsyncClient`
- **Import**: `get_embedding`
- **Variable**: `api_key`
  - **Type_alias**: `api_key`
- **Variable**: `api_key_path`
  - **Type_alias**: `api_key_path`
- **Variable**: `api_base`
  - **Type_alias**: `api_base`
- **Variable**: `verify_ssl_certs`
  - **Type_alias**: `verify_ssl_certs`
> *# No effect. Certificates are always verified.*
- **Variable**: `proxy`
  - **Type_alias**: `proxy`
- **Variable**: `app_info`
  - **Type_alias**: `app_info`
- **Variable**: `debug`
  - **Type_alias**: `debug`
- **Variable**: `log`
  - **Type_alias**: `log`
> *# Set to either 'debug' or 'info', controls console logging*
- **Variable**: `requestssession`
  - **Type_alias**: `requestssession`
> *# Provide a requests.Session or Session factory.*
- **Variable**: `aiosession`
  - **Type_alias**: `aiosession`
> *# Acts as a global aiohttp ClientSession that reuses connections.*
> *# This is user-supplied; otherwise, a session is remade for each request.*

---

### 2. /_base.py
- **Import**: `base64`
- **Import**: `io`
- **Import**: `json`
- **Import**: `ABC`
- **Import**: `functools`
- **Import**: `warnings`
- **Import**: `Any`
- **Import**: `hf_hub_download`
- **Import**: `PIL.Image`
- **Import**: `voyageai`
- **Import**: `voyageai.error as error`
- **Import**: `MultimodalInputRequest`
- **Import**: `default_api_key`
- **Import**: `EmbeddingsObject`
- **Function**: `def _get_client_config(`
- **Class**: `class _BaseClient(ABC)`
  >
  > Voyage AI Client
  > 
  >     Args:
  >         api_key (str): Your API key.
  >         max_retries (int): Maximum number of retries if API call fails.
  >         timeout (float): Timeout in seconds.
  - **Method**: `def __init__(`
    - **Variable**: `self.api_key`
      - **Type_alias**: `self.api_key`
    - **Variable**: `self._params`
      - **Type_alias**: `self._params`
  - **Method**: `@abstractmethod    `
  - **Method**: `@abstractmethod`
  - **Method**: `@abstractmethod`
  - **Method**: `@functools.lru_cache()`
  - **Method**: `def tokenize(`
  - **Method**: `def count_tokens(`
    - **Variable**: `tokenized`
      - **Type_alias**: `tokenized`
  - **Method**: `def count_usage(`
    >
    > This method returns estimated usage metrics for the provided input.
    >         Currently, only multimodal models are supported. Image URL segments are not supported.
    > 
    >         Args:
    >             inputs (list): a list of inputs
    >             model (str): the name of the model to be used for inference
    > 
    >         Returns:
    >             a dict, with the following keys:
    >             - for multimodal models:
    >               - "text_tokens": the number of tokens represented by the text in the items in the input
    >               - "image_pixels": the number of pixels represented by the images in the items in the input
    >               - "total_tokens": the total number of tokens represented by the items in the input
    - **Variable**: `client_config`
      - **Type_alias**: `client_config`
    - **Variable**: `min_pixels`
      - **Type_alias**: `min_pixels`
    - **Variable**: `max_pixels`
      - **Type_alias**: `max_pixels`
    - **Variable**: `pixel_to_token_ratio`
      - **Type_alias**: `pixel_to_token_ratio`
    - **Variable**: `request`
      - **Type_alias**: `request`
    - **Variable**: `image_tokens, image_pixels, text_tokens`
      - **Type_alias**: `image_tokens, image_pixels, text_tokens`

---

### 3. /api_resources/__init__.py
- **Import**: `VoyageResponse`
- **Import**: `VoyageHttpResponse`
- **Import**: `APIResource`
- **Import**: `Embedding`
- **Import**: `Reranking`
- **Import**: `MultimodalEmbedding`

---

### 4. /api_resources/api_requestor.py
- **Import**: `asyncio`
- **Import**: `json`
- **Import**: `time`
- **Import**: `platform`
- **Import**: `threading`
- **Import**: `time`
- **Import**: `warnings`
- **Import**: `JSONDecodeError`
- **Import**: `AsyncContextManager`
- **Import**: `urlencode`
- **Import**: `aiohttp`
- **Import**: `requests`
- **Import**: `voyageai`
- **Import**: `error`
- **Constant**: `TIMEOUT_SECS`
  - **Constant**: `TIMEOUT_SECS`
- **Constant**: `MAX_SESSION_LIFETIME_SECS`
  - **Constant**: `MAX_SESSION_LIFETIME_SECS`
- **Constant**: `MAX_CONNECTION_RETRIES`
  - **Constant**: `MAX_CONNECTION_RETRIES`
> *# Has one attribute per thread, 'session'.*
- **Function**: `def _build_api_url(url, query)`
  - **Variable**: `scheme, netloc, path, base_query, fragment`
    - **Type_alias**: `scheme, netloc, path, base_query, fragment`
- **Function**: `def _requests_proxies_arg(proxy) -> Optional[Dict[str, str]]`
  >
  > Returns a value suitable for the 'proxies' argument to 'requests.request.
- **Function**: `def _aiohttp_proxies_arg(proxy) -> Optional[str]`
  >
  > Returns a value suitable for the 'proxies' argument to 'aiohttp.ClientSession.request.
- **Function**: `def _make_session() -> requests.Session`
  - **Variable**: `s`
    - **Type_alias**: `s`
  - **Variable**: `proxies`
    - **Type_alias**: `proxies`
- **Class**: `class VoyageHttpResponse`
  - **Method**: `def __init__(self, data, headers)`
    - **Variable**: `self._headers`
      - **Type_alias**: `self._headers`
    - **Variable**: `self.data`
      - **Type_alias**: `self.data`
  - **Property**: `request_id`
  - **Property**: `retry_after`
  - **Property**: `operation_location`
  - **Property**: `organization`
  - **Property**: `response_ms`
    - **Variable**: `h`
      - **Type_alias**: `h`
- **Class**: `class APIRequestor`
  - **Method**: `def __init__(`
    - **Variable**: `self.api_base`
      - **Type_alias**: `self.api_base`
    - **Variable**: `self.api_key`
      - **Type_alias**: `self.api_key`
  - **Method**: `def request(`
    - **Variable**: `result`
      - **Type_alias**: `result`
    - **Variable**: `resp`
      - **Type_alias**: `resp`
  - **Method**: `async def arequest(`
    - **Variable**: `ctx`
      - **Type_alias**: `ctx`
    - **Variable**: `session`
      - **Type_alias**: `session`
    - **Variable**: `result`
      - **Type_alias**: `result`
    > *# Close the request before exiting session context.*
  - **Method**: `def request_headers(`
    - **Variable**: `user_agent`
      - **Type_alias**: `user_agent`
    - **Variable**: `uname_without_node`
      - **Type_alias**: `uname_without_node`
    - **Variable**: `ua`
      - **Type_alias**: `ua`
    - **Variable**: `headers`
      - **Type_alias**: `headers`
  - **Method**: `def _validate_headers(`
    - **Variable**: `headers`
      - **Type_alias**: `headers`
    > *# NOTE: It is possible to do more validation of the headers, but a request could always*
    > *# be made to the API manually with invalid headers, so we need to handle them server side.*
  - **Method**: `def _prepare_request_raw(`
    - **Variable**: `abs_url`
      - **Type_alias**: `abs_url`
    - **Variable**: `headers`
      - **Type_alias**: `headers`
    - **Variable**: `data`
      - **Type_alias**: `data`
    - **Variable**: `headers`
      - **Type_alias**: `headers`
  - **Method**: `def request_raw(`
    - **Variable**: `abs_url, headers, data`
      - **Type_alias**: `abs_url, headers, data`
    > *# Don't read the whole stream for debug logging unless necessary.*
  - **Method**: `async def arequest_raw(`
    - **Variable**: `abs_url, headers, data`
      - **Type_alias**: `abs_url, headers, data`
    - **Variable**: `request_kwargs`
      - **Type_alias**: `request_kwargs`
  - **Method**: `def _interpret_response(self, result: requests.Response) -> VoyageHttpResponse`
    >
    > Returns the response(s) and a bool indicating whether it is a stream.
  - **Method**: `async def _interpret_async_response(`
    >
    > Returns the response(s) and a bool indicating whether it is a stream.
  - **Method**: `def _interpret_response_line(`
    - **Variable**: `resp`
      - **Type_alias**: `resp`
  - **Method**: `def handle_error_response(self, rbody, rcode, resp, rheaders)`
- **Class**: `class AioHTTPSession(AsyncContextManager)`
  - **Method**: `def __init__(self)`
    - **Variable**: `self._session`
      - **Type_alias**: `self._session`
    - **Variable**: `self._should_close_session`
      - **Type_alias**: `self._should_close_session`
  - **Method**: `async def __aenter__(self)`
    - **Variable**: `self._session`
      - **Type_alias**: `self._session`
  - **Method**: `async def __aexit__(self, exc_type, exc_value, traceback)`

---

### 5. /api_resources/api_resource.py
- **Import**: `api_requestor`
- **Import**: `VoyageResponse`
- **Class**: `class APIResource(VoyageResponse)`
  - **Method**: `@classmethod`
    > *# Namespaces are separated in object names with periods (.) and in URLs*
    > *# with forward slashes (/), so replace the former with the latter.*
    - **Variable**: `base`
      - **Type_alias**: `base`
    > *# type: ignore*
  - **Method**: `@classmethod`
    - **Variable**: `requestor`
      - **Type_alias**: `requestor`
    - **Variable**: `url`
      - **Type_alias**: `url`
    - **Variable**: `headers`
      - **Type_alias**: `headers`
  - **Method**: `@classmethod`
    - **Variable**: `requestor, url, params, headers`
      - **Type_alias**: `requestor, url, params, headers`
    - **Variable**: `response`
      - **Type_alias**: `response`
    - **Variable**: `obj`
      - **Type_alias**: `obj`
  - **Method**: `@classmethod`
    - **Variable**: `requestor, url, params, headers`
      - **Type_alias**: `requestor, url, params, headers`
    - **Variable**: `response`
      - **Type_alias**: `response`
    - **Variable**: `obj`
      - **Type_alias**: `obj`

---

### 6. /api_resources/embedding.py
- **Import**: `APIResource`
- **Import**: `decode_base64_embedding`
- **Class**: `class Embedding(APIResource)`
  - **Constant**: `OBJECT_NAME`
    - **Constant**: `OBJECT_NAME`
  - **Method**: `@classmethod`
    >
    > Creates a new embedding for the provided input and parameters.
    - **Variable**: `user_provided_encoding_format`
      - **Type_alias**: `user_provided_encoding_format`
    > *# If encoding format was not explicitly specified, we opaquely use base64 for performance*
    - **Variable**: `response`
      - **Type_alias**: `response`
    > *# If a user specifies base64, we'll just return the encoded string.*
    > *# This is only for the default case.*
  - **Method**: `@classmethod`
    >
    > Creates a new embedding for the provided input and parameters.
    - **Variable**: `user_provided_encoding_format`
      - **Type_alias**: `user_provided_encoding_format`
    > *# If encoding format was not explicitly specified, we opaquely use base64 for performance*
    - **Variable**: `response`
      - **Type_alias**: `response`
    > *# If a user specifies base64, we'll just return the encoded string.*
    > *# This is only for the default case.*

---

### 7. /api_resources/multimodal_embedding.py
- **Import**: `Embedding`
- **Class**: `class MultimodalEmbedding(Embedding)`
  - **Constant**: `OBJECT_NAME`
    - **Constant**: `OBJECT_NAME`

---

### 8. /api_resources/reranking.py
- **Import**: `APIResource`
- **Class**: `class Reranking(APIResource)`
  - **Constant**: `OBJECT_NAME`
    - **Constant**: `OBJECT_NAME`
  - **Method**: `@classmethod`
    >
    > Creates a new reranking for the provided input and parameters.
    - **Variable**: `response`
      - **Type_alias**: `response`
  - **Method**: `@classmethod`
    >
    > Creates a new reranking for the provided input and parameters.
    - **Variable**: `response`
      - **Type_alias**: `response`

---

### 9. /api_resources/response.py
- **Import**: `json`
- **Import**: `deepcopy`
- **Import**: `VoyageHttpResponse`
- **Class**: `class VoyageResponse(dict)`
  - **Method**: `def __init__(`
  - **Method**: `def __setattr__(self, k, v)`
    - **Variable**: `self[k]`
      - **Type_alias**: `self[k]`
  - **Method**: `def __getattr__(self, k)`
  - **Method**: `def __delattr__(self, k)`
  - **Method**: `def __setitem__(self, k, v)`
  - **Method**: `def __delitem__(self, k)`
  > *# Custom unpickling method that uses `update` to update the dictionary*
  > *# without calling __setitem__, which would fail if any value is an empty*
  > *# string*
  - **Method**: `def __setstate__(self, state)`
  > *# Custom pickling method to ensure the instance is pickled as a custom*
  > *# class and not as a dict, otherwise __setstate__ would not be called when*
  > *# unpickling.*
  - **Method**: `def __reduce__(self)`
    - **Variable**: `reduce_value`
      - **Type_alias**: `reduce_value`
  - **Method**: `@classmethod`
    - **Variable**: `instance`
      - **Type_alias**: `instance`
  - **Method**: `def refresh_from(`
    - **Variable**: `self._previous`
      - **Type_alias**: `self._previous`
  - **Method**: `def __repr__(self)`
    - **Variable**: `ident_parts`
      - **Type_alias**: `ident_parts`
    - **Variable**: `obj`
      - **Type_alias**: `obj`
    - **Variable**: `unicode_repr`
      - **Type_alias**: `unicode_repr`
  - **Method**: `def __str__(self)`
    - **Variable**: `obj`
      - **Type_alias**: `obj`
  - **Method**: `def to_dict(self)`
  - **Method**: `def to_dict_recursive(self)`
    - **Variable**: `d`
      - **Type_alias**: `d`
  > *# This class overrides __setitem__ to throw exceptions on inputs that it*
  > *# doesn't like. This can cause problems when we try to copy an object*
  > *# wholesale because some data that's returned from the API may not be valid*
  > *# if it was set to be set manually. Here we override the class' copy*
  > *# arguments so that we can bypass these possible exceptions on __setitem__.*
  - **Method**: `def __copy__(self)`
    - **Variable**: `copied`
      - **Type_alias**: `copied`
  > *# This class overrides __setitem__ to throw exceptions on inputs that it*
  > *# doesn't like. This can cause problems when we try to copy an object*
  > *# wholesale because some data that's returned from the API may not be valid*
  > *# if it was set to be set manually. Here we override the class' copy*
  > *# arguments so that we can bypass these possible exceptions on __setitem__.*
  - **Method**: `def __deepcopy__(self, memo)`
    - **Variable**: `copied`
      - **Type_alias**: `copied`
    - **Variable**: `memo[id(self)]`
      - **Type_alias**: `memo[id(self)]`
- **Function**: `def convert_to_voyage_response(resp)`
- **Function**: `def convert_to_dict(obj)`
  >
  > Converts a VoyageResponse back to a regular dict.
  > 
  >     Nested VoyageResponse are also converted back to regular dicts.
  > 
  >     :param obj: The VoyageResponse to convert.
  > 
  >     :returns: The VoyageResponse as a dict.

---

### 10. /client.py
- **Import**: `warnings`
- **Import**: `Any`
- **Import**: `Retrying`
- **Import**: `Image`
- **Import**: `voyageai`
- **Import**: `_BaseClient`
- **Import**: `voyageai.error as error`
- **Import**: `MultimodalInputRequest`
- **Import**: `EmbeddingsObject`
- **Class**: `class Client(_BaseClient)`
  >
  > Voyage AI Client
  > 
  >     Args:
  >         api_key (str): Your API key.
  >         max_retries (int): Maximum number of retries if API call fails.
  >         timeout (float): Timeout in seconds.
  - **Method**: `def __init__(`
    - **Variable**: `self.retry_controller`
      - **Type_alias**: `self.retry_controller`
  - **Method**: `def embed(`
    - **Variable**: `result`
      - **Type_alias**: `result`
  - **Method**: `def rerank(`
    - **Variable**: `result`
      - **Type_alias**: `result`
  - **Method**: `def multimodal_embed(`
    >
    > Generate multimodal embeddings for the provided inputs using the specified model.
    > 
    >         :param inputs: Either a list of dictionaries (each with 'content') or a list of lists containing strings and/or PIL images.
    >         :param model: The model identifier.
    >         :param input_type: Optional input type.
    >         :param truncation: Whether to apply truncation.
    >         :return: An instance of MultimodalEmbeddingsObject.
    - **Variable**: `result`
      - **Type_alias**: `result`

---

### 11. /client_async.py
- **Import**: `warnings`
- **Import**: `List`
- **Import**: `Image`
- **Import**: `AsyncRetrying`
- **Import**: `voyageai`
- **Import**: `_BaseClient`
- **Import**: `voyageai.error as error`
- **Import**: `MultimodalInputRequest`
- **Import**: `EmbeddingsObject`
- **Class**: `class AsyncClient(_BaseClient)`
  >
  > Voyage AI Async Client
  > 
  >     Args:
  >         api_key (str): Your API key.
  >         max_retries (int): Maximum number of retries if API call fails.
  >         timeout (float): Timeout in seconds.
  - **Method**: `def __init__(`
    - **Variable**: `self.retry_controller`
      - **Type_alias**: `self.retry_controller`
  - **Method**: `async def embed(`
    - **Variable**: `result`
      - **Type_alias**: `result`
  - **Method**: `async def rerank(`
    - **Variable**: `result`
      - **Type_alias**: `result`
  - **Method**: `async def multimodal_embed(`
    >
    > Generate multimodal embeddings asynchronously for the provided inputs using the specified model.
    > 
    >         :param inputs: Either a list of dictionaries (each with 'content') or a list of lists containing strings and/or PIL images.
    >         :param model: The model identifier.
    >         :param input_type: Optional input type.
    >         :param truncation: Whether to apply truncation.
    >         :return: An instance of MultimodalEmbeddingsObject.
    - **Variable**: `result`
      - **Type_alias**: `result`

---

### 12. /embeddings_utils.py
- **Import**: `asyncio`
- **Import**: `AsyncLimiter`
- **Import**: `warnings`
- **Import**: `List`
- **Import**: `AsyncExitStack`
- **Import**: `retry`
- **Import**: `ThreadPoolExecutor`
- **Import**: `voyageai`
- **Constant**: `MAX_BATCH_SIZE`
  - **Constant**: `MAX_BATCH_SIZE`
- **Constant**: `MAX_LIST_LENGTH`
  - **Constant**: `MAX_LIST_LENGTH`
- **Constant**: `DEFAULT_CONCURRENCE`
  - **Constant**: `DEFAULT_CONCURRENCE`
- **Constant**: `DEFAULT_RPM`
  - **Constant**: `DEFAULT_RPM`
- **Function**: `@retry(`
  >
  > Python wrapper for one Voyage API call.
  - **Variable**: `data`
    - **Type_alias**: `data`
- **Function**: `@retry(`
  >
  > Python wrapper for one async Voyage API call.
  - **Variable**: `semaphore`
    - **Type_alias**: `semaphore`
  - **Variable**: `rate_limit`
    - **Type_alias**: `rate_limit`
- **Function**: `def _check_input_type(input_type: Optional[str])`
- **Function**: `def get_embedding(`
  >
  > Get Voyage embedding for a text string.
  > 
  >     Args:
  >         text (str): A text string to be embed.
  >         model (str): Name of the model to use.
  >         input_type (str): Type of the input text. Defalut to None, meaning the type is unspecified.
  >             Other options include: "query", "document".
- **Function**: `def get_embeddings(`
  >
  > Get Voyage embedding for a list of text strings.
  > 
  >     Args:
  >         list_of_text (list): A list of text strings to embed.
  >         model (str): Name of the model to use.
  >         input_type (str): Type of the input text. Defalut to None, meaning the type is unspecified.
  >             Other options include: "query", "document".
  - **Variable**: `batches`
    - **Type_alias**: `batches`
  - **Variable**: `results`
    - **Type_alias**: `results`
- **Function**: `async def aget_embedding(`
  >
  > Get Voyage embedding for a text string (async).
  > 
  >     Args:
  >         text (str): A text string to be embed.
  >         model (str): Name of the model to use.
  >         input_type (str): Type of the input text. Defalut to None, meaning the type is unspecified.
  >             Other options include: "query", "document".
- **Function**: `async def aget_embeddings(`
  >
  > Get Voyage embedding for a list of text strings (async).
  > 
  >     Args:
  >         list_of_text (list): A list of text strings to embed.
  >         model (str): Name of the model to use.
  >         input_type (str): Type of the input text. Defalut to None, meaning the type is unspecified.
  >             Other options include: "query", "document".
  - **Variable**: `semaphore`
    - **Type_alias**: `semaphore`
  - **Variable**: `rate_limit`
    - **Type_alias**: `rate_limit`
  - **Variable**: `batches`
    - **Type_alias**: `batches`
  - **Variable**: `async_tasks`
    - **Type_alias**: `async_tasks`
  - **Variable**: `results`
    - **Type_alias**: `results`

---

### 13. /error.py
- **Import**: `voyageai`
- **Class**: `class VoyageError(Exception)`
  - **Method**: `def __init__(`
    - **Variable**: `self._message`
      - **Type_alias**: `self._message`
    - **Variable**: `self.http_body`
      - **Type_alias**: `self.http_body`
    - **Variable**: `self.http_status`
      - **Type_alias**: `self.http_status`
    - **Variable**: `self.json_body`
      - **Type_alias**: `self.json_body`
    - **Variable**: `self.headers`
      - **Type_alias**: `self.headers`
    - **Variable**: `self.code`
      - **Type_alias**: `self.code`
    - **Variable**: `self.request_id`
      - **Type_alias**: `self.request_id`
    - **Variable**: `self.error`
      - **Type_alias**: `self.error`
  - **Method**: `def __str__(self)`
    - **Variable**: `msg`
      - **Type_alias**: `msg`
  > *# Returns the underlying `Exception` (base class) message, which is usually*
  > *# the raw message returned by OpenAI's API. This was previously available*
  > *# in python2 via `error.message`. Unlike `str(error)`, it omits "Request*
  > *# req_..." from the beginning of the string.*
  - **Property**: `user_message`
  - **Method**: `def __repr__(self)`
  - **Method**: `def construct_error_object(self)`
- **Class**: `class APIError(VoyageError)`
- **Class**: `class TryAgain(VoyageError)`
- **Class**: `class Timeout(VoyageError)`
- **Class**: `class APIConnectionError(VoyageError)`
- **Class**: `class InvalidRequestError(VoyageError)`
- **Class**: `class MalformedRequestError(VoyageError)`
- **Class**: `class AuthenticationError(VoyageError)`
- **Class**: `class RateLimitError(VoyageError)`
- **Class**: `class ServerError(VoyageError)`
- **Class**: `class ServiceUnavailableError(VoyageError)`

---

### 14. /object/__init__.py
- **Import**: `EmbeddingsObject`
- **Import**: `RerankingObject`
- **Import**: `MultimodalEmbeddingsObject`

---

### 15. /object/embeddings.py
- **Import**: `List`
- **Import**: `VoyageResponse`
- **Class**: `class EmbeddingsObject`
  - **Method**: `def __init__(self, response: Optional[VoyageResponse] = None)`
    - **Variable**: `self.embeddings`
      - **Type_alias**: `self.embeddings`
    - **Variable**: `self.total_tokens`
      - **Type_alias**: `self.total_tokens`
  - **Method**: `def update(self, response: VoyageResponse)`

---

### 16. /object/multimodal_embeddings.py
- **Import**: `base64`
- **Import**: `PIL.Image`
- **Import**: `PIL.ImageFile`
- **Import**: `BytesIO`
- **Import**: `Enum`
- **Import**: `List`
- **Import**: `error`
- **Import**: `VoyageResponse`
- **Class**: `class MultimodalEmbeddingsObject`
  - **Method**: `def __init__(self, response: Optional[VoyageResponse] = None)`
    - **Variable**: `self.embeddings`
      - **Type_alias**: `self.embeddings`
    - **Variable**: `self.text_tokens`
      - **Type_alias**: `self.text_tokens`
    - **Variable**: `self.image_pixels`
      - **Type_alias**: `self.image_pixels`
    - **Variable**: `self.total_tokens`
      - **Type_alias**: `self.total_tokens`
  - **Method**: `def update(self, response: VoyageResponse)`
- **Class**: `class MultimodalInputSegmentType(str, Enum)`
  - **Constant**: `TEXT`
    - **Constant**: `TEXT`
  - **Constant**: `IMAGE_URL`
    - **Constant**: `IMAGE_URL`
  - **Constant**: `IMAGE_BASE64`
    - **Constant**: `IMAGE_BASE64`
  - **Method**: `def __str__(self)`
- **Class**: `class MultimodalInputSegmentText(BaseModel)`
  - **Variable**: `type`
    - **Type_alias**: `type`
  - **Variable**: `text`
    - **Type_alias**: `text`
  - **Class**: `class Config`
    - **Variable**: `extra`
      - **Type_alias**: `extra`
- **Class**: `class MultimodalInputSegmentImageURL(BaseModel)`
  - **Variable**: `type`
    - **Type_alias**: `type`
  - **Variable**: `image_url`
    - **Type_alias**: `image_url`
  - **Class**: `class Config`
    - **Variable**: `extra`
      - **Type_alias**: `extra`
- **Class**: `class MultimodalInputSegmentImageBase64(BaseModel)`
  - **Variable**: `type`
    - **Type_alias**: `type`
  - **Variable**: `image_base64`
    - **Type_alias**: `image_base64`
  - **Class**: `class Config`
    - **Variable**: `extra`
      - **Type_alias**: `extra`
- **Class**: `class MultimodalInput(BaseModel)`
  - **Variable**: `content`
    - **Type_alias**: `content`
- **Class**: `class MultimodalInputRequest(BaseModel)`
  - **Variable**: `inputs`
    - **Type_alias**: `inputs`
  - **Variable**: `model`
    - **Type_alias**: `model`
  - **Variable**: `input_type`
    - **Type_alias**: `input_type`
  - **Variable**: `truncation`
    - **Type_alias**: `truncation`
  - **Method**: `@classmethod`
    >
    > Create a MultimodalInputRequest from user inputs.
    > 
    >         :param inputs: Either a list of dictionaries (each with 'content') or a list of lists containing strings and/or PIL images.
    >         :param model: The model identifier.
    >         :param input_type: Optional input type.
    >         :param truncation: Whether to apply truncation.
    >         :return: An instance of MultimodalInputRequest.
    >         :raises error.InvalidRequestError: If input processing fails.
    - **Variable**: `multimodal_inputs`
      - **Type_alias**: `multimodal_inputs`
    - **Variable**: `first_input`
      - **Type_alias**: `first_input`
    > *# Ensure all inputs are of the same kind*
    > *# Process inputs based on their kind*
  - **Method**: `@classmethod`
    >
    > Process a dictionary input and convert it to a MultimodalInput instance.
    > 
    >         :param input_data: The input dictionary.
    >         :param idx: The index of the input in the list.
    >         :return: A MultimodalInput instance.
    >         :raises ValueError: If 'content' key is missing or invalid.
  - **Method**: `@classmethod`
    >
    > Process a list input and convert it to a MultimodalInput instance.
    > 
    >         :param input_list: The input list containing strings or PIL images.
    >         :param idx: The index of the input in the list.
    >         :return: A MultimodalInput instance.
    >         :raises ValueError: If list items are not strings or PIL images.
    - **Variable**: `segments`
      - **Type_alias**: `segments`
  - **Method**: `@staticmethod`
    >
    > Create a segment based on the type of the item.
    > 
    >         :param item: The item to create a segment from.
    >         :param item_idx: The index of the item in the list.
    >         :param input_idx: The index of the input in the main inputs list.
    >         :return: A MultimodalInputSegment instance.
    >         :raises ValueError: If the item type is unsupported.
  - **Method**: `@staticmethod`
    >
    > Convert a PIL Image to a Base64-encoded data URI.
    > 
    >         :param image: The PIL Image to convert.
    >         :param target_format: The format to save the image in.
    >         :param target_mime_type: The MIME type of the image.
    >         :return: A Base64-encoded data URI string.
    - **Variable**: `buffered`
      - **Type_alias**: `buffered`
    - **Variable**: `img_base64`
      - **Type_alias**: `img_base64`

---

### 17. /object/reranking.py
- **Import**: `namedtuple`
- **Import**: `List`
- **Import**: `VoyageResponse`
- **Variable**: `RerankingResult`
  - **Type_alias**: `RerankingResult`
- **Class**: `class RerankingObject`
  - **Method**: `def __init__(self, documents: List[str], response: VoyageResponse)`
    - **Variable**: `self.results`
      - **Type_alias**: `self.results`
    - **Variable**: `self.total_tokens`
      - **Type_alias**: `self.total_tokens`

---

### 18. /util.py
- **Import**: `base64`
- **Import**: `logging`
- **Import**: `os`
- **Import**: `re`
- **Import**: `sys`
- **Import**: `Optional`
- **Import**: `numpy as np`
- **Import**: `voyageai`
- **Constant**: `VOYAGE_LOG`
  - **Constant**: `VOYAGE_LOG`
- **Variable**: `logger`
  - **Type_alias**: `logger`
- **Variable**: `api_key_to_header`
  - **Type_alias**: `api_key_to_header`
- **Function**: `def _console_log_level()`
- **Function**: `def log_debug(message, **params)`
  - **Variable**: `msg`
    - **Type_alias**: `msg`
- **Function**: `def log_info(message, **params)`
  - **Variable**: `msg`
    - **Type_alias**: `msg`
- **Function**: `def log_warn(message, **params)`
  - **Variable**: `msg`
    - **Type_alias**: `msg`
- **Function**: `def logfmt(props)`
  - **Function**: `def fmt(key, val)`
    > *# Check if val is already a string to avoid re-encoding into ascii.*
    > *# key should already be a string*
- **Function**: `def default_api_key() -> str`
  - **Variable**: `api_key_path`
    - **Type_alias**: `api_key_path`
  - **Variable**: `api_key`
    - **Type_alias**: `api_key`
  > *# When api_key_path is specified, it overwrites api_key*
- **Function**: `def _resolve_numpy_dtype(dtype: Optional[str] = None) -> str`
  - **Variable**: `dtype_mapping`
    - **Type_alias**: `dtype_mapping`
- **Function**: `def decode_base64_embedding(embedding: str, dtype: Optional[str] = None) -> Union[List[float], List[int]]`
  - **Variable**: `arr`
    - **Type_alias**: `arr`

---

### 19. /version.py
- **Import**: `importlib.metadata`
- **Constant**: `VERSION`
  - **Constant**: `VERSION`