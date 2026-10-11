# Squid Providers

Squid Providers contains the model integrations used by Squid Core and Code Squid.

It handles connections to model servers and APIs, model discovery, streaming,
tool calls, token usage, and converting provider responses into the format
Squid Core expects.

## Supported providers

| Provider | Default endpoint |
| --- | --- |
| Ollama | `http://127.0.0.1:11434` |
| LM Studio | `http://127.0.0.1:1234/v1` |
| OpenRouter | `https://openrouter.ai/api/v1` |
| OpenAI-compatible servers | `http://127.0.0.1:8000/v1` |

The provider IDs used in configuration are `ollama`, `lm_studio`, `openrouter`,
and `openai_compatible`.

Support for tools, images, reasoning, and other model features depends on the
provider and model.

OpenRouter requires an API key for inference. Local providers do not require one
unless the server itself is configured to use authentication.

## Installation

Squid Providers requires Python 3.11 or newer and depends on Squid Core.

Install from PyPI:

```bash
pip install squid-providers
```

Squid Core and the required dependencies are installed automatically.

The model server or API you want to use must be set up separately.

## Usage

This example connects to Ollama and sends a simple request using a model that is
already installed:

```python
from squid_providers.catalog import build_provider, resolve_config
from squid_providers.settings import ProviderSettings


config = resolve_config("ollama", {})

settings = ProviderSettings(
    provider="ollama",
    model_name="YOUR_MODEL_ID",
    base_url=config["base_url"],
    api_key=config["api_key"],
    config=config,
)

provider = build_provider(settings)

response = provider.create_response(
    model=settings.model_name,
    messages=[{"role": "user", "content": "Say hello in one sentence."}],
    tools=[],
)

print(response.content)
```

`build_provider()` creates the provider connection. Calling
`create_response()` sends the request to the model.

Providers follow Squid Core's provider contract and can be passed directly into
Squid Core.

## Configuration and model discovery

`resolve_config(provider_id, overrides)` combines provider defaults with any
connection settings you supply.

Connection settings can include:

- `base_url`
- `api_key`
- `api_key_env`

Ollama also supports `request_timeout`.

OpenRouter uses `OPENROUTER_API_KEY` by default.

Squid Providers also includes helpers for checking providers and discovering
available models:

- `provider_catalog()` returns provider information.
- `configuration_status()` checks whether a provider is configured.
- `provider_ready(settings)` checks whether the configured service is available.
- `discover_models(settings)` returns available model IDs.
- `discover_model_records(settings)` returns additional model information when
  the provider makes it available.

Applications are responsible for choosing models and storing their own settings.

## Building with Squid Providers

Squid Providers is separate from Squid Core so model integrations do not have to
be built into the agent itself.

Squid Core handles the agent loop and tool execution. Squid Providers handles
the connections to the models.

For a deeper look at the package and its public interfaces, see
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Testing

Run the test suite with:

```powershell
python -m pytest tests
```

The tests use mocked providers and do not require live model inference.

## License

Squid Providers is licensed under the [MIT License](LICENSE).