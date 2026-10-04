"""Focused coverage for session-backed image reads and provider translation."""

from __future__ import annotations

import base64
import hashlib
import importlib
import json
from io import BytesIO
from pathlib import Path
from squid_core.model_response import ModelResponse
from types import SimpleNamespace

import pytest
from PIL import Image

from squid_core.hooks.redundant_read import RepeatedReadNudge
from squid_core.hooks.runner import ToolResultContext
from squid_core.hooks.too_many_reads import TooManyReadsNudge
from squid_core.model_adapter import ModelAdapter, ModelRequestError
from squid_core.image_content import VISION_UNAVAILABLE_MESSAGE
from squid_providers.openai_compatible import (
    OpenAICompatibleProvider,
    _redacted_trace_payload as redact_openai_trace,
)
from squid_core.settings import ToolResultTrimmingSettings
from squid_core.storage import SessionStore
from squid_core.tool_result_trimming import render_messages_for_model
from squid_core.tools import ToolResults
from squid_core.tools.builtins.read_file import execute as read_file
from squid_core.tools.executor import ToolExecutor


image_reader_module = importlib.import_module("squid_core.file_readers.image")
atomic_json_module = importlib.import_module("squid_core.storage.atomic_json")


def image_bytes(image_format: str, *, size: tuple[int, int] = (3, 2)) -> bytes:
    """Create one tiny in-memory fixture without committing binary assets."""

    stream = BytesIO()
    Image.new("RGB", size, color=(220, 30, 40)).save(stream, format=image_format)
    return stream.getvalue()


















def image_tool_turn(result: dict, *, call_id: str = "image_call") -> list[dict]:
    action = {"tool": "read_file", "tool_call_id": call_id, "args": {}}
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            ],
        },
        ToolResults().message_from_action_result(action, result),
    ]




class _FakeCompletions:
    def __init__(self) -> None:
        self.payload = None

    def create(self, **payload):
        self.payload = payload
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="seen", tool_calls=[]),
                    finish_reason="stop",
                )
            ],
            usage=None,
        )


def lm_studio_provider() -> tuple[OpenAICompatibleProvider, _FakeCompletions]:
    provider = OpenAICompatibleProvider.__new__(OpenAICompatibleProvider)
    provider.provider_name = "lm_studio"
    provider.temperature = None
    provider.reasoning_mode = None
    provider.reasoning_token_budget = None
    provider._image_capability_cache = {"vision-model": True}
    completions = _FakeCompletions()
    provider.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return provider, completions


def test_openai_image_materializes_after_request_observation(tmp_path: Path) -> None:
    store = SessionStore.create(tmp_path / "sessions", "openai")
    data = image_bytes("PNG")
    asset_id = store.snapshot_asset(data, ".png")
    internal = {
        "role": "user",
        "content": [
            {"type": "text", "text": "inspect"},
            {
                "type": "image_asset",
                "asset_id": asset_id,
                "mime_type": "image/png",
                "description": "[image: fixture.png]",
            },
        ],
    }
    provider, completions = lm_studio_provider()
    observed = []
    adapter = ModelAdapter(
        provider,
        "vision-model",
        provider_name="lm_studio",
        request_observer=observed.append,
    )
    adapter.set_asset_loader(store.load_asset)

    adapter.respond([internal], [])

    assert observed[0]["messages"] == [internal]
    data_url = completions.payload["messages"][0]["content"][1]["image_url"]["url"]
    assert base64.b64decode(data_url.split(",", 1)[1]) == data
    assert "base64" not in json.dumps(observed)
    assert "base64" not in json.dumps(redact_openai_trace(completions.payload))








def test_canonical_persistence_replay_and_logs_keep_only_asset_reference(
    tmp_path: Path,
) -> None:
    store = SessionStore.create(tmp_path / "sessions", "replay")
    source = tmp_path / "fixture.png"
    original = image_bytes("PNG")
    source.write_bytes(original)
    result = read_file(tmp_path, str(source), snapshot_asset=store.snapshot_asset)
    messages = image_tool_turn(result)
    for message in messages:
        store.append_message(message)
    store.append_compaction_event(
        {
            "type": "compaction_applied",
            "messages_before": len(messages),
            "messages_after": 3,
            "replay": {
                "version": 1,
                "protected_message_count": 0,
                "preserved_message_count": 2,
                "compacted_message": {
                    "role": "assistant",
                    "content": "# Compacted Working Context\nPreserved recent image read.",
                },
            },
        }
    )
    rendered = render_messages_for_model(
        store.load_messages(), ToolResultTrimmingSettings()
    )
    store.append_model_view({"model": "vision", "provider": "test", "messages": rendered, "tools": []})
    source.write_bytes(image_bytes("PNG", size=(8, 8)))

    assert any(
        isinstance(message.get("content"), list)
        and any(block.get("type") == "image_asset" for block in message["content"])
        for message in rendered
    )
    assert store.load_asset(result["image"]["asset_id"]) == original

    # A later compaction may remove the old image turn from effective context,
    # but its canonical descriptor and immutable asset remain available on disk.
    store.append_compaction_event(
        {
            "type": "compaction_applied",
            "messages_before": 3,
            "messages_after": 1,
            "replay": {
                "version": 1,
                "protected_message_count": 0,
                "preserved_message_count": 0,
                "compacted_message": {
                    "role": "assistant",
                    "content": "# Compacted Working Context\nOld image turn omitted.",
                },
            },
        }
    )
    compacted_render = render_messages_for_model(
        store.load_messages(), ToolResultTrimmingSettings()
    )
    assert "image_asset" not in json.dumps(compacted_render)
    assert store.load_asset(result["image"]["asset_id"]) == original
    for path in (store.jsonl_path, store.model_view_path, store.model_transcript_path):
        persisted = path.read_text(encoding="utf-8")
        assert "data:image" not in persisted
        assert base64.b64encode(original).decode("ascii") not in persisted
    assert "[image:" in store.model_transcript_path.read_text(encoding="utf-8")




def test_provider_failure_after_snapshot_leaves_replayable_asset(tmp_path: Path) -> None:
    store = SessionStore.create(tmp_path / "sessions", "transport failure")
    source = tmp_path / "fixture.png"
    original = image_bytes("PNG")
    source.write_bytes(original)
    result = read_file(tmp_path, str(source), snapshot_asset=store.snapshot_asset)
    messages = render_messages_for_model(
        image_tool_turn(result), ToolResultTrimmingSettings()
    )

    provider, completions = lm_studio_provider()

    def fail_transport(**_payload):
        raise OSError("provider disconnected")

    completions.create = fail_transport
    adapter = ModelAdapter(provider, "vision-model", provider_name="lm_studio")
    adapter.set_asset_loader(store.load_asset)

    with pytest.raises(ModelRequestError, match="provider disconnected"):
        adapter.respond(messages, [])
    assert store.load_asset(result["image"]["asset_id"]) == original


def test_missing_replay_asset_and_image_hooks_fail_safely(tmp_path: Path) -> None:
    store = SessionStore.create(tmp_path / "sessions", "missing")
    data = image_bytes("PNG")
    asset_id = store.snapshot_asset(data, ".png")
    (store.assets_dir / asset_id).unlink()
    provider, _completions = lm_studio_provider()
    adapter = ModelAdapter(provider, "vision-model", provider_name="lm_studio")
    adapter.set_asset_loader(store.load_asset)
    message = {
        "role": "user",
        "content": [{"type": "image_asset", "asset_id": asset_id, "mime_type": "image/png"}],
    }
    with pytest.raises(ModelRequestError, match="Missing session image asset"):
        adapter.respond([message], [])

    result = {
        "ok": True,
        "tool": "read_file",
        "path": str(tmp_path / "image.png"),
        "media_type": "image",
        "image": {},
    }
    context = ToolResultContext(
        session_id="session",
        repo_root=str(tmp_path),
        action={"tool": "read_file", "args": {"path": result["path"]}},
        result=result,
    )
    assert RepeatedReadNudge().tool_result(context) is None
    assert TooManyReadsNudge().tool_result(context) is None
