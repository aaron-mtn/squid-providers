# Squid Providers

Squid Providers supplies first-party model transports, provider configuration,
and model discovery for Squid Core. It translates native Ollama and
OpenAI-compatible chat-completions responses into Core's `ModelResponse`
contract, including streaming, tool calls, and token usage when available.
Applications select and configure a provider, then pass the constructed transport
to Squid Core, which owns the agent runtime and tool execution.

## Supported providers

| Provider ID | Backend | Default endpoint |
| --- | --- | --- |
| `ollama` | Native Ollama chat | `http://127.0.0.1:11434` |
| `lm_studio` | LM Studio chat completions | `http://127.0.0.1:1234/v1` |
| `openrouter` | OpenRouter chat completions | `https://openrouter.ai/api/v1` |
| `openai_compatible` | Other compatible chat-completions servers | `http://127.0.0.1:8000/v1` |
| `squid_models` | Squid Models managed transport profile | `http://127.0.0.1:8080/v1` |

Backend/model support for tools, images, and reasoning varies. OpenRouter
inference requires an API key; its model catalog is public. Local compatible
profiles supply placeholder keys by default; configure a real key if your server
requires one. Ollama has no credential fields. The application-facing
`provider_catalog()` excludes the managed Squid Models entry so applications can
compose their own product entry.

## Installation

Python 3.11 or newer is required. Version 0.1.0 depends on `squid-core==0.1.0`,
`openai`, and `httpx`. This first GitHub release does not assume PyPI availability.

Replace `CORE_OWNER` and `PROVIDERS_OWNER` below with the actual GitHub owners.
The commands require a `v0.1.0` tag in both repositories; these are release URL
placeholders until the repositories and tags are available. Git must be installed.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install "squid-core @ git+https://github.com/CORE_OWNER/squid-core.git@v0.1.0" "squid-providers @ git+https://github.com/PROVIDERS_OWNER/squid-providers.git@v0.1.0"
```

Alternatively, install local checkouts together (the Core checkout must be
version 0.1.0):

```powershell
python -m pip install ../squid-core .
```

This explicit Core source avoids looking up an unpublished dependency on PyPI.
The sibling layout is a convenience for local development, not a runtime
requirement. Provider services and models must be set up separately.

## Usage

With Ollama running and a model already installed, use its exact model ID in
place of `YOUR_MODEL_ID`:

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

`build_provider()` constructs a transport; the response call performs inference.
Import the public APIs from their concrete modules; the package root does not
re-export them. Transports also expose `stream_chat()` and implement Core's
`BaseProvider` contract for injection into a Core runtime.

## Discovery and configuration

`resolve_config(provider_id, overrides)` merges sparse connection overrides with
provider defaults and resolves credential environment variables. Supported
connection fields are `base_url`, `api_key`, and `api_key_env`; Ollama additionally
supports a positive, finite `request_timeout`. OpenRouter defaults to
`OPENROUTER_API_KEY`. Applications store overrides under `providers.<id>`;
pass that provider's dictionary to `resolve_config()`. Sampling and reasoning
controls belong on `ProviderSettings`, rather than in the connection dictionary.

`provider_catalog()` returns non-secret external provider metadata.
`configuration_status()` checks configuration without network requests;
`provider_ready(settings)` probes the running service.
`discover_models(settings)` returns sorted, deduplicated exact model IDs, while
`discover_model_records(settings)` also returns available display names and load
state. Discovery uses bounded HTTP requests and provider-specific catalog
fallbacks; failures can raise errors. Applications own model selection, labels,
and persistence. Discovery does not perform inference.

## Development and tests

For the supplied editable development setup, clone Squid Core and Squid Providers
as siblings, create and activate a virtual environment, then run from this repo:

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest tests
python -m compileall -q squid_providers
git diff --check
```

Tests use mocked transports and require no live model inference. Run other
repositories' suites separately to avoid duplicate test module names.

To build and inspect release artifacts:

```powershell
python -m pip install build
python -m build
python -m zipfile -l dist/squid_providers-0.1.0-py3-none-any.whl
python -m tarfile -l dist/squid_providers-0.1.0.tar.gz
```

The wheel contains the runtime package, metadata, and license. The source
distribution also includes README and architecture documentation; tests, private
development logs, and local build/cache files are excluded.
See [Architecture](docs/ARCHITECTURE.md) for ownership, public seams, and runtime
invariants.

## License

MIT, copyright Aaron Slater. See [LICENSE](LICENSE).
