"""Verify function strictness at the serialized Chat Completions boundary."""

from copy import deepcopy
import json

import httpx
from openai import OpenAI
import pytest

from squid_core.tools.builtins.read_file import tool_definition
from squid_providers.openai_compatible import OpenAICompatibleProvider


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("provider_name", ["openrouter", "openai_compatible"])
@pytest.mark.parametrize("strict", ["omitted", False, True])
def test_serialized_function_strictness_preserves_schema_and_source(
    monkeypatch, streaming: bool, provider_name: str, strict: object,
) -> None:
    """Exercise both SDK paths without contacting a model or altering schemas."""
    tools = [tool_definition()]
    if strict != "omitted":
        tools[0]["function"]["strict"] = strict
    original = deepcopy(tools)
    requests = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if streaming:
            response = {
                "id": "fixture", "object": "chat.completion.chunk",
                "created": 0, "model": "fixture",
                "choices": [{"index": 0, "delta": {"content": "done"},
                             "finish_reason": "stop"}],
            }
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"},
                text=f"data: {json.dumps(response)}\n\ndata: [DONE]\n\n",
            )
        return httpx.Response(200, json={
            "id": "fixture", "object": "chat.completion", "created": 0,
            "model": "fixture",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "done"}}],
        })

    def make_client(**kwargs):
        return OpenAI(
            **kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handle)),
        )

    # Replace only SDK construction: serialization and stream decoding stay real.
    monkeypatch.setattr("squid_providers.openai_compatible.OpenAI", make_client)
    provider = OpenAICompatibleProvider(
        "https://fixture.invalid/v1", "test-only", provider_name=provider_name,
    )
    try:
        if streaming:
            result = provider.stream_chat(
                model="fixture", messages=[], tools=tools,
                on_text_delta=lambda _text: None,
            )
        else:
            result = provider.create_response(model="fixture", messages=[], tools=tools)
    finally:
        provider.client.close()

    expected = deepcopy(original)
    if provider_name == "openrouter" and strict == "omitted":
        expected[0]["function"]["strict"] = False
    assert result.content == "done"
    assert len(requests) == 1
    assert requests[0].get("stream", False) is streaming
    assert requests[0]["tools"] == expected
    assert requests[0]["tools"][0]["function"]["parameters"]["required"] == ["path"]
    assert tools == original


def test_openrouter_leaves_non_function_tools_unchanged() -> None:
    """Strictness applies only to function tools, not other tool contracts."""
    tools = [{"type": "custom", "custom": {"name": "fixture"}}]
    original = deepcopy(tools)
    provider = OpenAICompatibleProvider(
        "https://fixture.invalid/v1", None, provider_name="openrouter",
    )
    payload = provider._payload(model="fixture", messages=[], tools=tools)
    assert payload["tools"] == original
    assert tools == original
