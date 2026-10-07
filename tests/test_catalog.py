"""Provider catalog, construction, discovery and replay policy regressions."""

import json
import pytest
from squid_providers.catalog import PROVIDER_PROFILES, build_provider, tool_call_content_mode
from squid_providers.settings import ProviderSettings
from squid_providers.ollama import DEFAULT_OLLAMA_REQUEST_TIMEOUT, OllamaProvider
from squid_providers.openai_compatible import OpenAICompatibleProvider

class JsonResponse:
    """Fake response for provider discovery without network requests."""
    def __init__(self, payload): self.payload = payload
    def __enter__(self): return self
    def __exit__(self, *_args): return None
    def read(self): return json.dumps(self.payload).encode()

def test_friendly_profiles_map_to_existing_transports() -> None:
    profiles = {profile.friendly_name: profile for profile in PROVIDER_PROFILES}
    assert list(profiles) == [
        "Squid Models",
        "LM Studio",
        "Ollama",
        "OpenRouter",
        "OpenAI Compatible",
    ]
    for friendly_name in ("Squid Models", "LM Studio", "OpenRouter", "OpenAI Compatible"):
        profile = profiles[friendly_name]
        provider = build_provider(
            ProviderSettings(
                provider=profile.provider_id,
                model_name="selected-model",
                base_url=profile.base_url,
                api_key=profile.api_key or "test-key",
                temperature=None,
                reasoning_mode=None,
            )
        )
        assert isinstance(provider, OpenAICompatibleProvider)
        assert provider.provider_name == profile.provider_id
    ollama = profiles["Ollama"]
    provider = build_provider(
        ProviderSettings(
            provider=ollama.provider_id,
            model_name="selected-model",
            base_url=ollama.base_url,
            api_key=None,
            temperature=None,
            reasoning_mode=None,
        )
    )
    assert isinstance(provider, OllamaProvider)


def test_lm_studio_native_discovery_and_loaded_readiness(monkeypatch) -> None:
    requests = []

    def urlopen(http_request, timeout):
        requests.append((http_request, timeout))
        return JsonResponse({
            "models": [
                {"type": "embedding", "key": "ignore"},
                {"type": "llm", "key": "model-a", "loaded_instances": [{}]},
                {"type": "llm", "key": "model-b", "loaded_instances": []},
            ]
        })

    monkeypatch.setattr("squid_providers.openai_compatible.request.urlopen", urlopen)
    provider = OpenAICompatibleProvider(
        "http://127.0.0.1:1234/v1", "lm-studio", provider_name="lm_studio"
    )
    assert provider.is_ready() is True
    assert provider.list_models() == ["model-a", "model-b"]
    assert all(item[0].full_url.endswith("/api/v1/models") for item in requests)


def test_squid_models_health_and_real_model_discovery(monkeypatch) -> None:
    requests = []

    def urlopen(http_request, timeout):
        requests.append(http_request)
        if http_request.full_url.endswith("/health"):
            return JsonResponse({"status": "ok"})
        return JsonResponse({"data": [{"id": "squid-a"}, {"id": "squid-b"}]})

    monkeypatch.setattr("squid_providers.openai_compatible.request.urlopen", urlopen)
    provider = OpenAICompatibleProvider(
        "http://127.0.0.1:8080/v1", "local", provider_name="squid_models"
    )
    assert provider.is_ready() is True
    assert provider.list_models() == ["squid-a", "squid-b"]
    assert [item.full_url for item in requests] == [
        "http://127.0.0.1:8080/health",
        "http://127.0.0.1:8080/v1/models",
    ]


def test_ollama_native_discovery(monkeypatch) -> None:
    requests = []

    def urlopen(http_request, timeout):
        requests.append(http_request)
        return JsonResponse({"models": [{"name": "qwen:latest"}, {"model": "gemma:2b"}]})

    monkeypatch.setattr("squid_providers.ollama.request.urlopen", urlopen)
    provider = OllamaProvider()
    assert provider.is_ready() is True
    assert provider.list_models() == ["qwen:latest", "gemma:2b"]
    assert all(item.full_url.endswith("/api/tags") for item in requests)


def test_openrouter_inference_construction_still_requires_key() -> None:
    settings = ProviderSettings(
        provider="openrouter",
        model_name="vendor/model",
        base_url="https://openrouter.ai/api/v1",
        api_key=None,
        temperature=None,
        reasoning_mode=None,
    )

    with pytest.raises(ValueError, match="api_key is required"):
        build_provider(settings)


def test_provider_selection_defaults_to_openai_compatible() -> None:
    settings = ProviderSettings(
        provider="openai_compatible",
        model_name="model",
        base_url="http://127.0.0.1:1234/v1",
        api_key="key",
    )

    provider = build_provider(settings)

    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.temperature is None


def test_openai_compatible_construction_requires_effective_api_key() -> None:
    settings = ProviderSettings(
        provider="openai_compatible",
        model_name="model",
        base_url="http://127.0.0.1:1234/v1",
        api_key=None,
    )

    with pytest.raises(ValueError, match="api_key"):
        build_provider(settings)


def test_provider_selection_uses_native_ollama_when_configured() -> None:
    settings = ProviderSettings(
        provider="ollama",
        model_name="model",
        base_url="http://127.0.0.1:11434/v1",
        api_key=None,
    )

    provider = build_provider(settings)

    assert isinstance(provider, OllamaProvider)
    assert provider.base_url == "http://127.0.0.1:11434"
    assert provider.temperature is None


def test_provider_selection_uses_native_ollama_default_base_url() -> None:
    settings = ProviderSettings(
        provider="ollama",
        model_name="model",
        base_url=None,
        api_key=None,
    )

    provider = build_provider(settings)

    assert isinstance(provider, OllamaProvider)
    assert provider.base_url == "http://127.0.0.1:11434"
    assert provider.request_timeout == DEFAULT_OLLAMA_REQUEST_TIMEOUT


def test_provider_selection_threads_configured_timeout_to_ollama() -> None:
    settings = ProviderSettings(
        provider="ollama",
        model_name="model",
        base_url=None,
        api_key=None,
        config={"request_timeout": 45},
    )

    provider = build_provider(settings)

    assert isinstance(provider, OllamaProvider)
    assert provider.request_timeout == 45.0


def test_provider_selection_threads_temperature() -> None:
    settings = ProviderSettings(
        provider="openai_compatible",
        model_name="model",
        base_url="http://127.0.0.1:1234/v1",
        api_key="key",
        temperature=0,
    )

    provider = build_provider(settings)

    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.temperature == 0


def test_provider_selection_uses_openai_compatible_for_openrouter() -> None:
    settings = ProviderSettings(
        provider="openrouter",
        model_name="model",
        base_url="https://openrouter.ai/api/v1",
        api_key="key",
        temperature=0,
        reasoning_mode="medium",
    )

    provider = build_provider(settings)

    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.temperature == 0
    assert provider.reasoning_mode == "medium"
    assert provider.provider_name == "openrouter"


def test_provider_selection_preserves_managed_llama_cpp_identity() -> None:
    settings = ProviderSettings(
        provider="squid_models",
        model_name="model",
        base_url="http://127.0.0.1:1234/v1",
        api_key="key",
    )

    provider = build_provider(settings)

    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.provider_name == "squid_models"


def test_openrouter_uses_null_tool_call_content_mode() -> None:
    settings = ProviderSettings(
        provider="openrouter",
        model_name="model",
        base_url="https://openrouter.ai/api/v1",
        api_key="key",
    )

    assert tool_call_content_mode(settings) == "null"


def test_openai_compatible_preserves_default_tool_call_content_mode() -> None:
    settings = ProviderSettings(
        provider="openai_compatible",
        model_name="model",
        base_url="http://127.0.0.1:1234/v1",
        api_key="key",
    )

    assert tool_call_content_mode(settings) == "empty"


def test_provider_selection_threads_temperature_to_ollama() -> None:
    settings = ProviderSettings(
        provider="ollama",
        model_name="model",
        base_url=None,
        api_key=None,
        temperature=0,
    )

    provider = build_provider(settings)

    assert isinstance(provider, OllamaProvider)
    assert provider.temperature == 0



@pytest.mark.parametrize("value", [0, -1, True, False, "45", [], float("inf"), float("nan")])
def test_invalid_scoped_timeout_is_rejected(value):
    from squid_providers.catalog import validate_config
    with pytest.raises(ValueError, match="request_timeout"):
        validate_config("ollama", {"request_timeout": value})


@pytest.mark.parametrize("api_key", ["configured-token", None])
def test_lm_studio_metadata_uses_configured_authentication(monkeypatch, api_key):
    requests = []

    def urlopen(http_request, timeout):
        requests.append(http_request)
        return JsonResponse({"models": [{
            "type": "llm", "key": "vision", "loaded_instances": [{"id": "loaded"}],
            "capabilities": {"vision": True},
        }]})

    monkeypatch.setattr("squid_providers.openai_compatible.request.urlopen", urlopen)
    provider = OpenAICompatibleProvider("http://fixture/v1", api_key, provider_name="lm_studio")
    assert provider.list_models() == ["vision"]
    assert provider.is_ready() is True
    assert provider.supports_image_input("vision") is True
    assert len(requests) == 3
    assert all(req.get_header("Authorization") == (
        f"Bearer {api_key}" if api_key else None
    ) for req in requests)


def test_lm_studio_embedding_only_catalog_is_not_chat_ready(monkeypatch):
    monkeypatch.setattr("squid_providers.openai_compatible.request.urlopen", lambda *_a, **_k:
        JsonResponse({"models": [{"type": "embedding", "key": "embed", "loaded_instances": [{}]}]}))
    provider = OpenAICompatibleProvider("http://fixture/v1", None, provider_name="lm_studio")
    assert provider.list_models() == []
    assert provider.is_ready() is False


def test_external_catalog_is_safe_and_separates_provider_identities(monkeypatch):
    from dataclasses import replace
    from squid_providers import catalog
    monkeypatch.setenv("OPENROUTER_API_KEY", "environment-secret")
    monkeypatch.setattr(catalog, "PROVIDER_PROFILES", tuple(
        replace(profile, api_key="profile-secret") for profile in catalog.PROVIDER_PROFILES
    ))
    records = catalog.provider_catalog()
    assert {row["id"]: row["label"] for row in records} == {
        "lm_studio": "LM Studio", "ollama": "Ollama",
        "openrouter": "OpenRouter", "openai_compatible": "OpenAI Compatible",
    }
    assert "secret" not in json.dumps(records)
    assert all("api_key" not in row["defaults"] for row in records)
    assert records[1]["connection_fields"] == ["base_url"]
    assert records[1]["config_fields"] == ["request_timeout"]
    records[0]["defaults"]["base_url"] = "mutated"
    assert catalog.provider_catalog()[0]["defaults"]["base_url"] != "mutated"


def test_connection_configuration_and_status_are_provider_owned():
    from squid_providers import catalog
    router = catalog.normalize_connection_config("openrouter")
    assert router["api_key_env"] == "OPENROUTER_API_KEY"
    assert not catalog.configuration_status("openrouter", router, environment={})["configured"]
    assert catalog.configuration_status("openrouter", router, environment={
        "OPENROUTER_API_KEY": "secret",
    }) == {"configured": True, "credentials_configured": True}
    saved = catalog.normalize_connection_config("openrouter", {"api_key": "secret"}, router)
    assert catalog.normalize_connection_config("openrouter", {"api_key": ""}, saved) == saved
    ollama = catalog.normalize_connection_config("ollama", {"api_key": "discard", "api_key_env": "DISCARD"},
                                                 {"request_timeout": 42})
    assert ollama["api_key"] == ollama["api_key_env"] == ""
    assert ollama["request_timeout"] == 42
    assert catalog.configuration_status("ollama", ollama, environment={})["configured"]
    assert catalog.normalize_connection_config("lm_studio")["base_url"] == "http://127.0.0.1:1234/v1"
    assert catalog.normalize_connection_config("openai_compatible")["base_url"] == "http://127.0.0.1:8000/v1"
