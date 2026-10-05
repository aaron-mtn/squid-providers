"""Provider-owned vision discovery and native wire encoding."""

import base64
import json

import pytest

from squid_core.model_adapter import ModelAdapter
from squid_core.image_content import VISION_UNAVAILABLE_MESSAGE
from squid_providers.ollama import (
    OllamaProvider, convert_messages, _redacted_trace_payload as redact_ollama_trace,
)
from squid_providers.openai_compatible import OpenAICompatibleProvider


def test_lm_studio_capability_is_read_from_model_metadata() -> None:
    provider = OpenAICompatibleProvider("http://fixture/v1", None, provider_name="lm_studio")
    provider._lm_studio_models = lambda: [
        {"key": "vision", "loaded_instances": [], "capabilities": {"vision": True}},
        {"key": "text", "loaded_instances": [], "capabilities": {"vision": False}},
    ]

    assert provider.supports_image_input("vision") is True
    assert provider.supports_image_input("text") is False

    adapter = ModelAdapter(provider, "text", provider_name="lm_studio")
    request = adapter.prepare_request(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_asset",
                        "asset_id": "asset.png",
                        "mime_type": "image/png",
                    }
                ],
            }
        ],
        [],
    )
    assert "image_asset" not in json.dumps(request)
    assert VISION_UNAVAILABLE_MESSAGE in json.dumps(request)


def test_generic_openai_provider_fails_closed_for_images() -> None:
    provider = OpenAICompatibleProvider("http://fixture/v1", None)
    adapter = ModelAdapter(provider, "unknown", provider_name="openai_compatible")
    adapter.set_asset_loader(lambda _asset_id: b"data")
    message = {
        "role": "user",
        "content": [{"type": "image_asset", "asset_id": "x", "mime_type": "image/png"}],
    }

    request = adapter.prepare_request([message], [])
    assert "image_asset" not in json.dumps(request)
    assert VISION_UNAVAILABLE_MESSAGE in json.dumps(request)


def test_ollama_translation_and_capability(monkeypatch: pytest.MonkeyPatch) -> None:
    # Wire conversion preserves bytes; image validation is a Core asset concern.
    data = b"snapshot bytes"
    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": "inspect"},
            {"type": "image_asset", "asset_id": "asset.png", "mime_type": "image/png"},
        ],
    }
    converted = convert_messages([message], asset_loader=lambda _asset_id: data)
    assert converted[0]["content"] == "inspect"
    assert base64.b64decode(converted[0]["images"][0]) == data
    assert "base64" not in json.dumps(redact_ollama_trace({"messages": converted}))

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return b'{"capabilities":["completion","vision"]}'

    monkeypatch.setattr("squid_providers.ollama.request.urlopen", lambda *_a, **_k: Response())
    assert OllamaProvider().supports_image_input("vision-model") is True

    class TextResponse(Response):
        def read(self):
            return b'{"capabilities":["completion"]}'

    monkeypatch.setattr(
        "squid_providers.ollama.request.urlopen",
        lambda *_a, **_k: TextResponse(),
    )
    text_provider = OllamaProvider()
    assert text_provider.supports_image_input("text-model") is False
    adapter = ModelAdapter(text_provider, "text-model", provider_name="ollama")
    request = adapter.prepare_request([message], [])
    assert "image_asset" not in json.dumps(request)
    assert VISION_UNAVAILABLE_MESSAGE in json.dumps(request)
