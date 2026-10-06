"""Normalized inventory contracts, independent of app labels and selection."""

import json
from urllib.error import HTTPError, URLError

import pytest

from squid_providers.catalog import ModelRecord, discover_model_records, discover_models
from squid_providers.settings import ProviderSettings
from squid_providers.openai_compatible import OpenAICompatibleProvider
from squid_providers.ollama import OllamaProvider


class JsonResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        pass

    def read(self):
        return json.dumps(self.payload).encode()


def settings(provider, api_key=None, config=None):
    return ProviderSettings(provider, "configured/exact:ID", "http://fixture/v1",
                            api_key, config=config or {})


@pytest.mark.parametrize("provider", ["openrouter", "ollama", "openai_compatible", "lm_studio", "squid_models"])
def test_inventory_order_deduplication_and_exact_case(monkeypatch, provider):
    ids = ["z/Model:Q4", "alpha", "Alpha", "Beta", "alpha", "", None, 7]
    calls = []

    def urlopen(req, timeout):
        calls.append(req)
        if req.full_url.endswith("/api/v1/models"):
            raise URLError("not native")
        field, id_field = ("models", "name") if provider == "ollama" else ("data", "id")
        return JsonResponse({field: [{id_field: model_id} for model_id in ids] + [None]})

    monkeypatch.setattr("squid_providers.discovery.request.urlopen", urlopen)
    expected = ["Alpha", "alpha", "Beta", "z/Model:Q4"]
    assert discover_model_records(settings(provider)) == [ModelRecord(model_id) for model_id in expected]
    assert discover_models(settings(provider)) == expected
    assert calls[-1].full_url == ("http://fixture/api/tags" if provider == "ollama" else "http://fixture/v1/models")


def test_native_metadata_is_alphabetical_without_app_grouping(monkeypatch):
    monkeypatch.setattr("squid_providers.discovery.request.urlopen", lambda *_a, **_k:
        JsonResponse({"models": [
            {"type": "llm", "key": "z", "display_name": " Zed ", "loaded_instances": [{}]},
            {"type": "llm", "key": "a", "loaded_instances": []},
            {"type": "llm", "key": "z", "loaded_instances": []},
            {"type": "llm", "key": "a", "display_name": "Alpha", "loaded_instances": [{}]},
            {"type": "llm", "key": "A", "loaded_instances": "invalid"},
            {"type": "embedding", "key": "embed", "loaded_instances": [{}]},
            {"type": "llm", "key": None},
        ]}))
    assert discover_model_records(settings("lm_studio")) == [
        ModelRecord("A", loaded=False), ModelRecord("a", "Alpha", True), ModelRecord("z", "Zed", True),
    ]


@pytest.mark.parametrize("native", [None, {}, {"models": []}, {"models": [{"type": "embedding", "key": "embed"}]}, {"models": [{"type": "llm", "key": ""}]}])
@pytest.mark.parametrize("provider", ["lm_studio", "openai_compatible"])
def test_unusable_native_inventory_falls_back(monkeypatch, native, provider):
    urls = []

    def urlopen(req, timeout):
        urls.append(req.full_url)
        return JsonResponse(native if len(urls) == 1 else {"data": [{"id": "fallback"}]})

    monkeypatch.setattr("squid_providers.discovery.request.urlopen", urlopen)
    assert discover_models(settings(provider)) == ["fallback"]
    assert urls == ["http://fixture/api/v1/models", "http://fixture/v1/models"]


@pytest.mark.parametrize("error", [URLError("offline"), HTTPError("http://fixture", 404, "missing", {}, None), TimeoutError()])
def test_native_request_errors_fall_back(monkeypatch, error):
    def urlopen(req, timeout):
        if req.full_url.endswith("/api/v1/models"):
            raise error
        return JsonResponse({"data": []})

    monkeypatch.setattr("squid_providers.discovery.request.urlopen", urlopen)
    assert discover_models(settings("lm_studio")) == []


@pytest.mark.parametrize("provider", ["openrouter", "ollama", "squid_models"])
def test_empty_inventory_and_malformed_list(monkeypatch, provider):
    field = "models" if provider == "ollama" else "data"
    monkeypatch.setattr("squid_providers.discovery.request.urlopen", lambda *_a, **_k: JsonResponse({field: []}))
    assert discover_models(settings(provider)) == []
    monkeypatch.setattr("squid_providers.discovery.request.urlopen", lambda *_a, **_k: JsonResponse({field: {}}))
    with pytest.raises(RuntimeError, match="model list"):
        discover_models(settings(provider))


@pytest.mark.parametrize("provider", ["openrouter", "lm_studio"])
def test_display_names_and_credentials(monkeypatch, provider):
    calls = []
    monkeypatch.setenv("CATALOG_KEY", "secret")

    def urlopen(req, timeout):
        calls.append(req)
        if provider == "lm_studio":
            return JsonResponse({"models": [{"type": "llm", "key": "id", "display_name": "Friendly", "loaded_instances": []}]})
        return JsonResponse({"data": [{"id": "id", "name": "Friendly"}]})

    monkeypatch.setattr("squid_providers.discovery.request.urlopen", urlopen)
    records = discover_model_records(settings(provider, config={"api_key_env": "CATALOG_KEY"}))
    assert records[0].display_name == "Friendly"
    assert calls[0].get_header("Authorization") == "Bearer secret"


def test_ollama_alternate_model_field(monkeypatch):
    monkeypatch.setattr("squid_providers.discovery.request.urlopen", lambda *_a, **_k:
        JsonResponse({"models": [{"model": "exact:Tag"}]}))
    assert discover_models(settings("ollama")) == ["exact:Tag"]


def test_ids_are_opaque_and_absent_loaded_metadata_is_unknown(monkeypatch):
    monkeypatch.setattr("squid_providers.discovery.request.urlopen", lambda *_a, **_k:
        JsonResponse({"models": [
            {"type": "llm", "key": " Model:Q4 "},
            {"type": "llm", "key": "Model:Q4"},
            {"type": "llm", "key": "   "},
        ]}))
    assert discover_model_records(settings("lm_studio")) == [
        ModelRecord(" Model:Q4 "), ModelRecord("Model:Q4"),
    ]


@pytest.mark.parametrize("provider", ["openai_compatible", "ollama", "lm_studio"])
def test_raw_transport_keeps_native_order(monkeypatch, provider):
    ids = ["z", "a", "A"]
    if provider == "ollama":
        payload = {"models": [{"name": name} for name in ids]}
        transport = OllamaProvider("http://fixture/v1")
    elif provider == "lm_studio":
        payload = {"models": [{"key": name, "type": "llm"} for name in ids]}
        transport = OpenAICompatibleProvider("http://fixture/v1", None, provider_name=provider)
    else:
        payload = {"data": [{"id": name} for name in ids]}
        transport = OpenAICompatibleProvider("http://fixture/v1", None)
    monkeypatch.setattr("squid_providers.discovery.request.urlopen", lambda *_a, **_k: JsonResponse(payload))
    assert transport.list_models() == ids
