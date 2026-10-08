"""Pin OpenRouter catalog capabilities through Core's image-read/request seams.

All catalogs and completions are local doubles; these tests own no live model
calls. Core still owns image validation, asset persistence and text read semantics.
"""

import base64
import json
from io import BytesIO
from types import SimpleNamespace

import pytest
from PIL import Image

from squid_core.image_content import VISION_UNAVAILABLE_MESSAGE
from squid_core.model_adapter import ModelAdapter
from squid_core.session import Session
from squid_core.settings import ToolResultTrimmingSettings
from squid_core.storage import SessionStore
from squid_core.tool_result_trimming import render_messages_for_model
from squid_core.tools import ToolResults
from squid_core.tools.executor import ToolExecutor
from squid_providers import discovery
from squid_providers.catalog import build_provider, tool_call_content_mode
from squid_providers.openai_compatible import OpenAICompatibleProvider
from squid_providers.settings import ProviderSettings


VISION_MODEL = "deepseek/deepseek-v4.1-flash"
TEXT_MODEL = "deepseek/deepseek-chat-v3-0324"
BASE_URL = "http://openrouter-fixture/v1"
# Schema fixtures, not a captured live API response. Capability decisions must
# follow these advertised fields rather than special-casing either model slug.
CATALOG = {"data": [
    {"id": VISION_MODEL, "architecture": {"input_modalities": ["text", "image"]}},
    {"id": TEXT_MODEL, "architecture": {"input_modalities": ["text"]}},
]}


@pytest.fixture(autouse=True)
def isolated_catalog(monkeypatch):
    # A process cache must never let another test's metadata affect these guards.
    monkeypatch.setattr(discovery, "_openrouter_image_capabilities", {})


def provider():
    return OpenAICompatibleProvider(BASE_URL, None, provider_name="openrouter")


def test_transport_catalog_capability_is_exact_and_cached(monkeypatch):
    transport = provider()
    calls = []

    def catalog(url, *, authenticated=True):
        calls.append((url, authenticated))
        return CATALOG

    monkeypatch.setattr(transport, "_get_json", catalog)
    assert transport.supports_image_input(VISION_MODEL) is True
    assert transport.supports_image_input(TEXT_MODEL) is False
    assert transport.supports_image_input(VISION_MODEL.upper()) is False
    assert transport.supports_image_input("unknown-vision-model") is False
    assert transport.list_models() == [VISION_MODEL, TEXT_MODEL]
    assert calls == [(BASE_URL + "/models", False)] * 2


def test_inventory_metadata_is_reused_by_new_runtime_provider(monkeypatch):
    monkeypatch.setattr(discovery, "_get_json", lambda *_a, **_k: CATALOG)
    settings = ProviderSettings("openrouter", VISION_MODEL, BASE_URL, "fixture-key")
    records = discovery.discover_model_records(settings)
    assert {record.model_id for record in records} == {VISION_MODEL, TEXT_MODEL}
    transport = build_provider(settings)

    def unexpected_request(*_args, **_kwargs):
        pytest.fail("fresh UI catalog must not cause a second metadata request")

    monkeypatch.setattr(transport, "_get_json", unexpected_request)
    assert ModelAdapter(transport, VISION_MODEL).supports_image_input() is True
    assert ModelAdapter(transport, TEXT_MODEL).supports_image_input() is False


def test_arbitrary_catalog_image_model_is_supported_without_name_guessing(monkeypatch):
    transport = provider()
    monkeypatch.setattr(transport, "_get_json", lambda *_a, **_k: {
        "data": [{"id": "vendor/opaque-id", "architecture": {
            "input_modalities": ["image", "text"],
        }}],
    })
    assert transport.supports_image_input("vendor/opaque-id") is True


@pytest.mark.parametrize("architecture", [
    None, {}, {"input_modalities": None}, {"input_modalities": "text,image"},
    {"input_modalities": ["text"], "output_modalities": ["image"]},
    {"modality": "text+image->text"},
])
def test_missing_or_malformed_input_modalities_fail_closed(monkeypatch, architecture):
    transport = provider()
    monkeypatch.setattr(transport, "_get_json", lambda *_a, **_k: {
        "data": [{"id": VISION_MODEL, "architecture": architecture}, None],
    })
    assert transport.supports_image_input(VISION_MODEL) is False


def test_capability_cache_expires_and_is_endpoint_scoped(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(discovery.time, "monotonic", lambda: now[0])
    discovery.remember_openrouter_capabilities(BASE_URL + "/", CATALOG["data"])
    assert discovery.cached_openrouter_capabilities(BASE_URL)[VISION_MODEL] is True
    assert discovery.cached_openrouter_capabilities("http://other/v1") is None
    now[0] = discovery.OPENROUTER_CAPABILITY_TTL_SECONDS
    transport = provider()
    monkeypatch.setattr(transport, "_get_json", lambda *_a, **_k: {
        "data": [{"id": VISION_MODEL, "architecture": {"input_modalities": ["text"]}}],
    })
    assert transport.supports_image_input(VISION_MODEL) is False


@pytest.mark.parametrize("payload", [None, {}, {"data": {}}])
def test_unavailable_catalog_fails_closed_in_adapter(monkeypatch, payload):
    transport = provider()
    monkeypatch.setattr(transport, "_get_json", lambda *_a, **_k: payload)
    assert ModelAdapter(transport, VISION_MODEL).supports_image_input() is False


def test_catalog_connection_failure_fails_closed_in_adapter(monkeypatch):
    transport = provider()

    def offline(*_args, **_kwargs):
        raise OSError("offline")

    monkeypatch.setattr(transport, "_get_json", offline)
    assert ModelAdapter(transport, VISION_MODEL).supports_image_input() is False


@pytest.mark.parametrize("model", [VISION_MODEL, TEXT_MODEL])
def test_read_file_image_reaches_multimodal_transport_only_for_vision(
    monkeypatch, tmp_path, model,
):
    transport = provider()
    monkeypatch.setattr(transport, "_get_json", lambda *_a, **_k: CATALOG)
    payloads = []

    def complete(**payload):
        payloads.append(payload)
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content="seen", tool_calls=[]), finish_reason="stop",
        )], usage=None)

    transport.client = SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=complete),
    ))
    settings = ProviderSettings("openrouter", model, BASE_URL, None)
    adapter = ModelAdapter(transport, model, provider_name="openrouter",
                           tool_call_content_mode=tool_call_content_mode(settings))
    store = SessionStore.create(tmp_path / "sessions", "image-read")
    adapter.set_asset_loader(store.load_asset)
    executor = ToolExecutor(tmp_path, Session(tmp_path),
                            snapshot_asset=store.snapshot_asset,
                            supports_image_input=adapter.supports_image_input)
    stream = BytesIO()
    Image.new("RGB", (3, 2), color="red").save(stream, format="PNG")
    data = stream.getvalue()
    source = tmp_path / "fixture.png"
    source.write_bytes(data)
    action = {"tool": "read_file", "tool_call_id": "read-image",
              "args": {"path": str(source)}}
    result = executor.execute(action)

    if model == TEXT_MODEL:
        assert result["ok"] is False
        assert result["error"] == VISION_UNAVAILABLE_MESSAGE
        assert not store.assets_dir.exists()
        assert payloads == []
        return

    assert result["ok"] is True
    messages = [
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "read-image", "type": "function", "function": {
                "name": "read_file", "arguments": json.dumps(action["args"]),
            },
        }]},
        ToolResults().message_from_action_result(action, result),
    ]
    adapter.respond(render_messages_for_model(messages, ToolResultTrimmingSettings()), [])
    assert payloads[0]["model"] == VISION_MODEL
    image_urls = [block["image_url"]["url"]
                  for message in payloads[0]["messages"]
                  if isinstance(message.get("content"), list)
                  for block in message["content"] if block.get("type") == "image_url"]
    assert len(image_urls) == 1
    assert base64.b64decode(image_urls[0].split(",", 1)[1]) == data


@pytest.mark.parametrize("model", [VISION_MODEL, TEXT_MODEL])
def test_text_read_does_not_probe_model_capabilities(monkeypatch, tmp_path, model):
    transport = provider()

    def unexpected_request(*_args, **_kwargs):
        pytest.fail("ordinary text reads must not request model metadata")

    monkeypatch.setattr(transport, "_get_json", unexpected_request)
    adapter = ModelAdapter(transport, model)
    executor = ToolExecutor(tmp_path, Session(tmp_path),
                            supports_image_input=adapter.supports_image_input)
    source = tmp_path / "notes.txt"
    source.write_text("hello\nworld\n", encoding="utf-8")
    result = executor.execute({"tool": "read_file", "args": {"path": str(source)}})
    assert result["ok"] is True
    assert result["content"] == "hello\nworld\n"
    assert result["content_truncated"] is False
    assert result["total_lines"] == 2
