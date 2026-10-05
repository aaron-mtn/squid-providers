"""Tests for native Ollama transport, adaptation, and transcript replay."""

from squid_core.model_response import ModelResponse, TokenUsage, ToolCall
import json
from io import BytesIO
import httpx
from urllib.error import HTTPError, URLError

import pytest

from squid_core.model_adapter import (
    ModelAdapter,
    ModelRequestError,
    model_request_error_details,
)
from squid_core.providers.base import StreamCancelled
from squid_providers.ollama import (
    DEFAULT_OLLAMA_REQUEST_TIMEOUT,
    OllamaProvider,
    adapt_response,
    convert_messages,
)
from squid_core.settings import ToolResultTrimmingSettings
from squid_core.tool_result_trimming import render_messages_for_model


class CapturingOllamaProvider(OllamaProvider):
    def __init__(self, temperature=None) -> None:
        super().__init__(temperature=temperature)
        self.payloads = []

    def _post_chat(self, payload: dict) -> dict:
        self.payloads.append(payload)
        return {"message": {"role": "assistant", "content": "Done."}}


class RespondingOllamaProvider(OllamaProvider):
    """Return a fixed sequence of native responses without network access."""

    def __init__(self, responses: list[dict]) -> None:
        super().__init__()
        self.responses = iter(responses)

    def _post_chat(self, payload: dict) -> dict:
        return next(self.responses)


class StreamingOllamaProvider(OllamaProvider):
    def __init__(self, chunks: list[dict]) -> None:
        super().__init__()
        self.chunks = chunks
        self.payloads = []

    def _post_chat_stream(self, payload: dict):
        self.payloads.append(payload)
        yield from self.chunks


class ClosableStream:
    """Small iterator double that records the provider cleanup contract."""

    def __init__(self, chunks: list[dict]) -> None:
        self._chunks = iter(chunks)
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._chunks)

    def close(self) -> None:
        self.closed = True


class ClosableStreamingOllamaProvider(OllamaProvider):
    def __init__(self, stream: ClosableStream) -> None:
        super().__init__()
        self.stream = stream

    def _post_chat_stream(self, payload: dict):
        return self.stream


class FakeHTTPResponse:
    """Minimal context-managed HTTP response for transport-focused tests."""

    def __init__(self, body: bytes, lines: list[bytes] | None = None) -> None:
        self.body = body
        self.lines = lines or []

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def read(self) -> bytes:
        return self.body

    def __iter__(self):
        return iter(self.lines)


def test_configured_timeout_is_passed_to_normal_and_streaming_transports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    responses = iter(
        [
            FakeHTTPResponse(b'{"message":{"content":"Done."}}'),
            FakeHTTPResponse(
                b"",
                [b'{"message":{"content":"Streamed."},"done":true}\n'],
            ),
        ]
    )

    def urlopen(http_request, *, timeout):
        calls.append((http_request.full_url, timeout))
        return next(responses)

    monkeypatch.setattr("squid_providers.ollama.request.urlopen", urlopen)
    provider = OllamaProvider(request_timeout=12.5)

    normal = provider.create_response("test-model", [], [])
    streamed = provider.stream_chat("test-model", [], [], lambda _delta: None)

    assert calls == [
        ("http://127.0.0.1:11434/api/chat", 12.5),
        ("http://127.0.0.1:11434/api/chat", 12.5),
    ]
    assert isinstance(normal, ModelResponse)
    assert isinstance(streamed, ModelResponse)
    assert normal.content == "Done."
    assert streamed.content == "Streamed."


def test_default_timeout_is_finite_and_provider_owned() -> None:
    provider = OllamaProvider()

    assert provider.request_timeout == DEFAULT_OLLAMA_REQUEST_TIMEOUT


@pytest.mark.parametrize("streaming", [False, True], ids=["normal", "streaming"])
def test_timeout_uses_model_request_failure_path(
    monkeypatch: pytest.MonkeyPatch,
    streaming: bool,
) -> None:
    def urlopen(_http_request, *, timeout):
        assert timeout == 4.0
        raise TimeoutError("Ollama stalled")

    monkeypatch.setattr("squid_providers.ollama.request.urlopen", urlopen)
    adapter = ModelAdapter(
        OllamaProvider(request_timeout=4),
        model_name="test-model",
    )

    with pytest.raises(ModelRequestError, match="Ollama stalled") as exc_info:
        if streaming:
            adapter.respond_streaming([], [], lambda _delta: None)
        else:
            adapter.respond([], [])

    assert isinstance(exc_info.value.__cause__, TimeoutError)


def test_nested_urllib_timeout_remains_a_transient_timeout_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def urlopen(_http_request, *, timeout):
        raise URLError(TimeoutError(f"stalled after {timeout} seconds"))

    monkeypatch.setattr("squid_providers.ollama.request.urlopen", urlopen)
    adapter = ModelAdapter(OllamaProvider(request_timeout=3), "test-model")

    with pytest.raises(ModelRequestError) as exc_info:
        adapter.respond([], [])

    details = model_request_error_details(exc_info.value)
    assert details["failure_class"] == "timeout"
    assert details["_transient"] is True


def test_stream_trace_reports_native_raw_thinking_and_summary() -> None:
    chunks = [
        {"id": "ollama_trace", "message": {"thinking": "thinking"}},
        {
            "message": {"content": "Ready."},
            "done": True,
            "done_reason": "stop",
        },
    ]
    provider = StreamingOllamaProvider(chunks)
    events = []

    provider.stream_chat(
        model="test-model",
        messages=[{"role": "user", "content": "prompt"}],
        tools=[],
        on_text_delta=lambda _delta: None,
        trace_callback=events.append,
    )

    assert [event["type"] for event in events] == [
        "request_start", "raw_chunk", "raw_chunk", "request_complete",
    ]
    assert events[1]["raw"] is chunks[0]
    assert events[1]["thinking_delta"] == "thinking"
    assert events[2]["content_delta"] == "Ready."
    assert events[-1]["response_id"] == "ollama_trace"
    assert events[-1]["finish_reason"] == "stop"
    assert events[-1]["first_raw_chunk_elapsed"] is not None
    assert events[-1]["first_reasoning_delta_elapsed"] is not None
    assert events[-1]["first_visible_content_delta_elapsed"] is not None
    assert events[-1]["total_reasoning_characters"] == len("thinking")
    assert events[-1]["total_visible_content_characters"] == len("Ready.")
    assert events[-1]["total_raw_chunk_count"] == 2


def test_native_stream_forwards_normalized_reasoning_deltas() -> None:
    provider = StreamingOllamaProvider(
        [
            {"message": {"thinking": "first "}},
            {
                "message": {"thinking": "second"},
                "done": True,
                "done_reason": "stop",
            },
        ]
    )
    reasoning_deltas = []

    provider.stream_chat(
        model="test-model",
        messages=[],
        tools=[],
        on_text_delta=lambda _delta: None,
        on_reasoning_delta=reasoning_deltas.append,
    )

    assert reasoning_deltas == ["first ", "second"]


def test_assistant_tool_call_arguments_are_converted_to_dict() -> None:
    converted = convert_messages(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path": "/tmp/input.py"}',
                        },
                    }
                ],
            }
        ]
    )

    assert converted == [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": {"path": "/tmp/input.py"},
                    },
                }
            ],
        }
    ]


def test_tool_result_uses_tool_name_from_previous_assistant_call() -> None:
    converted = convert_messages(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path": "/tmp/input.py"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": '{"ok": true}',
            },
        ]
    )

    assert converted[1] == {
        "role": "tool",
        "tool_name": "read_file",
        "content": '{"ok": true}',
    }


def test_unknown_tool_result_id_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="unknown tool_call_id: call_missing"):
        convert_messages(
            [
                {
                    "role": "tool",
                    "tool_call_id": "call_missing",
                    "content": "{}",
                }
            ]
        )


def test_native_response_with_content_adapts_to_generic_shape() -> None:
    response = adapt_response({"message": {"role": "assistant", "content": "Done."}})

    assert response.content == "Done."
    assert response.tool_calls == []


def test_native_response_preserves_available_thinking_finish_reason_and_id() -> None:
    response = adapt_response(
        {
            "id": "ollama_response_1",
            "message": {
                "role": "assistant",
                "content": "Done.",
                "thinking": "provider-exposed thinking",
            },
            "done_reason": "stop",
        }
    )

    assert response.response_id == "ollama_response_1"
    assert response.finish_reason == "stop"
    assert response.reasoning == "provider-exposed thinking"
    assert isinstance(response, ModelResponse)


def test_native_tool_calls_get_generated_ids_and_json_string_arguments() -> None:
    response = adapt_response(
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "read_file",
                            "arguments": {"path": "/tmp/input.py"},
                        }
                    }
                ],
            }
        }
    )

    tool_call = response.tool_calls[0]

    assert tool_call.id.startswith("ollama_call_")
    assert tool_call.id.endswith("_0")
    assert tool_call.name == "read_file"
    assert json.loads(tool_call.arguments) == {"path": "/tmp/input.py"}


def test_missing_ids_are_unique_across_non_streaming_responses() -> None:
    native_response = {
        "message": {
            "role": "assistant",
            "tool_calls": [
                {"function": {"name": "read_file", "arguments": {}}},
            ],
        }
    }
    provider = RespondingOllamaProvider([native_response, native_response])

    first = provider.create_response("test-model", [], [])
    second = provider.create_response("test-model", [], [])

    first_id = first.tool_calls[0].id
    second_id = second.tool_calls[0].id
    assert first_id != second_id


def test_multiple_missing_ids_share_a_response_prefix_but_not_an_id() -> None:
    response = adapt_response(
        {
            "message": {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"name": "read_file", "arguments": {}}},
                    {"function": {"name": "run_command", "arguments": {}}},
                ],
            }
        }
    )

    first, second = response.tool_calls
    assert first.id != second.id
    assert first.id.rsplit("_", 1)[0] == second.id.rsplit("_", 1)[0]
    assert first.id.endswith("_0")
    assert second.id.endswith("_1")


def test_provider_supplied_tool_call_id_is_preserved_exactly() -> None:
    response = adapt_response(
        {
            "message": {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "provider-call-123",
                        "function": {"name": "read_file", "arguments": {}},
                    }
                ],
            }
        }
    )

    assert response.tool_calls[0].id == "provider-call-123"


def test_temperature_none_omits_options_temperature_from_payload() -> None:
    provider = CapturingOllamaProvider(temperature=None)

    provider.create_response(model="test-model", messages=[], tools=[])

    payload = provider.payloads[0]
    assert "options" not in payload


def test_native_ollama_thinking_control_is_request_scoped() -> None:
    provider = CapturingOllamaProvider()

    provider.create_response(model="test-model", messages=[], tools=[])
    provider.create_response(
        model="test-model",
        messages=[],
        tools=[],
        please_stop_contemplating_the_void=True,
    )

    assert "think" not in provider.payloads[0]
    assert provider.payloads[1]["think"] is False


def test_native_ollama_streaming_thinking_control_is_request_scoped() -> None:
    provider = StreamingOllamaProvider([
        {"message": {"content": "Done."}, "done": True},
    ])

    provider.stream_chat(
        model="test-model",
        messages=[],
        tools=[],
        on_text_delta=lambda _delta: None,
        please_stop_contemplating_the_void=True,
    )

    assert provider.payloads[0]["think"] is False


def test_zero_temperature_is_included_in_options_payload() -> None:
    provider = CapturingOllamaProvider(temperature=0)

    provider.create_response(model="test-model", messages=[], tools=[])

    assert provider.payloads[0]["options"]["temperature"] == 0


def test_native_ollama_payload_does_not_include_parallel_tool_calls() -> None:
    provider = CapturingOllamaProvider()

    provider.create_response(
        model="test-model",
        messages=[],
        tools=[{"type": "function", "function": {"name": "read_file"}}],
    )

    assert "parallel_tool_calls" not in provider.payloads[0]


def test_native_ollama_streaming_accumulates_text_and_tool_calls_once() -> None:
    provider = StreamingOllamaProvider(
        [
            {"message": {"role": "assistant", "content": "I will "}, "done": False},
            {"message": {"role": "assistant", "content": "read the file."}, "done": False},
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "read_file",
                                "arguments": {"path": "/tmp/example.py"},
                            }
                        }
                    ],
                },
                "done": False,
            },
            {"done": True, "done_reason": "stop"},
        ]
    )
    adapter = ModelAdapter(provider, model_name="test-model")
    deltas = []
    preparing_events = []

    reply = adapter.respond_streaming(
        messages=[],
        tool_definitions=[
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "parameters": {
                        "type": "object",
                        "required": ["path"],
                        "properties": {"path": {"type": "string"}},
                    },
                },
            }
        ],
        on_text_delta=deltas.append,
        on_tool_preparing=preparing_events.append,
    )

    assert provider.payloads[0]["stream"] is True
    assert deltas == ["I will ", "read the file."]
    assert reply["message"]["content"] == "I will read the file."
    call_id = reply["message"]["tool_calls"][0]["id"]
    assert call_id.startswith("ollama_call_")
    assert call_id.endswith("_0")
    assert reply["actions"] == [
        {
            "tool": "read_file",
            "args": {"path": "/tmp/example.py"},
            "tool_call_id": call_id,
            "raw_arguments": '{"path": "/tmp/example.py"}',
        }
    ]
    assert reply["message"]["tool_calls"] == [
        {
            "id": call_id,
            "type": "function",
            "function": {
                "name": "read_file",
                "arguments": '{"path": "/tmp/example.py"}',
            },
        }
    ]
    assert preparing_events == [{"tool": "read_file", "tool_call_index": 0}]


def test_streamed_missing_ids_are_unique_across_responses() -> None:
    provider = StreamingOllamaProvider(
        [
            {
                "message": {
                    "tool_calls": [
                        {"function": {"name": "read_file", "arguments": {}}},
                    ]
                },
                "done": True,
            }
        ]
    )

    first = provider.stream_chat("test-model", [], [], lambda _delta: None)
    second = provider.stream_chat("test-model", [], [], lambda _delta: None)

    first_id = first.tool_calls[0].id
    second_id = second.tool_calls[0].id
    assert first_id != second_id


def test_distinct_fallback_ids_preserve_historical_rendering_and_conversion() -> None:
    provider = RespondingOllamaProvider(
        [
            {
                "message": {
                    "tool_calls": [
                        {"function": {"name": "read_file", "arguments": {}}},
                    ]
                }
            },
            {
                "message": {
                    "tool_calls": [
                        {"function": {"name": "run_command", "arguments": {}}},
                    ]
                }
            },
        ]
    )
    adapter = ModelAdapter(provider, model_name="test-model")
    definitions = [
        {
            "type": "function",
            "function": {
                "name": name,
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in ("read_file", "run_command")
    ]

    first = adapter.respond([], definitions)
    second = adapter.respond([], definitions)
    first_id = first["actions"][0]["tool_call_id"]
    second_id = second["actions"][0]["tool_call_id"]
    messages = [
        first["message"],
        {
            "role": "tool",
            "tool_call_id": first_id,
            "content": json.dumps(
                {"ok": True, "tool": "read_file", "content": "old source"}
            ),
        },
        second["message"],
        {
            "role": "tool",
            "tool_call_id": second_id,
            "content": json.dumps(
                {"ok": True, "tool": "run_command", "output": "new output"}
            ),
        },
    ]

    rendered = render_messages_for_model(
        messages,
        ToolResultTrimmingSettings(
            recent_full_results=0,
            read_old_content_chars=1,
            command_old_output_chars=1,
        ),
    )
    converted = convert_messages(rendered)

    assert first_id != second_id
    assert rendered[0]["tool_calls"][0]["id"] == first_id
    assert rendered[1]["tool_call_id"] == first_id
    assert rendered[1]["content"].startswith("READ_FILE OK")
    assert rendered[2]["tool_calls"][0]["id"] == second_id
    assert rendered[3]["tool_call_id"] == second_id
    assert rendered[3]["content"].startswith("RUN_COMMAND OK")
    assert converted[1]["tool_name"] == "read_file"
    assert converted[3]["tool_name"] == "run_command"


@pytest.mark.parametrize(
    "chunks",
    [
        [{"message": {"content": "partial"}, "done": False}],
        [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "read_file",
                                "arguments": {"path": "/tmp/incomplete"},
                            }
                        }
                    ]
                },
                "done": False,
            }
        ],
        [],
    ],
    ids=["partial-text", "partial-tool-call", "empty"],
)
def test_native_ollama_incomplete_stream_fails_and_closes(chunks) -> None:
    stream = ClosableStream(chunks)
    provider = ClosableStreamingOllamaProvider(stream)

    with pytest.raises(RuntimeError, match="ended before provider signaled completion"):
        provider.stream_chat(
            model="test-model",
            messages=[],
            tools=[],
            on_text_delta=lambda _delta: None,
        )

    assert stream.closed is True


def test_native_ollama_completed_stream_closes() -> None:
    stream = ClosableStream(
        [{"message": {"content": "complete"}, "done": True, "done_reason": "stop"}]
    )
    provider = ClosableStreamingOllamaProvider(stream)

    response = provider.stream_chat(
        model="test-model",
        messages=[],
        tools=[],
        on_text_delta=lambda _delta: None,
    )

    assert response.content == "complete"
    assert stream.closed is True


def test_native_ollama_stream_cancellation_remains_distinct_and_closes() -> None:
    stream = ClosableStream(
        [
            {"message": {"content": "first"}, "done": False},
            {"message": {"content": "second"}, "done": False},
        ]
    )
    provider = ClosableStreamingOllamaProvider(stream)
    cancelled = False

    def on_delta(_delta: str) -> None:
        nonlocal cancelled
        cancelled = True

    with pytest.raises(StreamCancelled):
        provider.stream_chat(
            model="test-model",
            messages=[],
            tools=[],
            on_text_delta=on_delta,
            should_cancel=lambda: cancelled,
        )

    assert stream.closed is True


@pytest.mark.parametrize("counts, expected", [
    ({}, None),
    ({"prompt_eval_count": 0}, TokenUsage(0, None, None)),
    ({"eval_count": 2}, TokenUsage(None, 2, None)),
    ({"prompt_eval_count": 5, "eval_count": 2}, TokenUsage(5, 2, 7)),
])
def test_native_usage_normalization_preserves_missing_counts(counts, expected):
    assert adapt_response({"message": {"content": ""}, **counts}).usage == expected


@pytest.mark.parametrize("aliases, expected", [
    ({"reasoning": "first", "reasoning_content": "second", "thinking": "third"}, "first"),
    ({"reasoning": "", "reasoning_content": "second", "thinking": "third"}, "second"),
    ({"thinking": "third"}, "third"),
])
def test_native_reasoning_precedence(aliases, expected):
    assert adapt_response({"message": aliases}).reasoning == expected


def test_native_stream_normalizes_metadata_and_keeps_malformed_arguments(monkeypatch):
    provider = OllamaProvider()
    monkeypatch.setattr(provider, "_post_chat_stream", lambda _payload: (item for item in [
        {"id": "stream-id", "message": {"reasoning_content": "secondary", "reasoning": "primary", "tool_calls": [
            {"id": "a", "function": {"name": "read_file", "arguments": '{"path":'}},
            {"id": "b", "function": {"name": "read_file", "arguments": '{}'}},
        ]}},
        {"done": True, "done_reason": "stop", "prompt_eval_count": 5, "eval_count": 2},
    ]))
    response = provider.stream_chat("test-model", [], [], lambda _delta: None)
    assert isinstance(response, ModelResponse)
    assert response.reasoning == "primary"
    assert response.usage == TokenUsage(5, 2, 7)
    assert response.response_id == "stream-id"
    assert response.finish_reason == "stop"
    assert response.tool_calls == [ToolCall("a", "read_file", '{"path":'), ToolCall("b", "read_file", '{}')]


@pytest.mark.parametrize("groups", [[1, 1, 1], [2, 1], [3]])
def test_streamed_complete_calls_preserve_arrival_order_and_ids(groups):
    native_calls = [
        {"id": "provided-a", "function": {"index": 0, "name": "read_file", "arguments": {"path": "a"}}},
        {"function": {"index": 1, "name": "read_file", "arguments": {"path": "b"}}},
        {"id": "provided-c", "function": {"index": 2, "name": "other_tool", "arguments": {"value": 3}}},
    ]
    chunks = []
    offset = 0
    for size in groups:
        chunks.append({"message": {"tool_calls": native_calls[offset:offset + size]}})
        offset += size
    chunks.append({"done": True, "done_reason": "stop"})
    prepared = []
    response = StreamingOllamaProvider(chunks).stream_chat(
        "model", [], [], lambda _delta: None, on_tool_preparing=prepared.append,
    )
    assert [call.name for call in response.tool_calls] == ["read_file", "read_file", "other_tool"]
    assert [json.loads(call.arguments) for call in response.tool_calls] == [{"path": "a"}, {"path": "b"}, {"value": 3}]
    assert response.tool_calls[0].id == "provided-a"
    assert response.tool_calls[2].id == "provided-c"
    assert [event["tool_call_index"] for event in prepared] == [0, 1, 2]


@pytest.mark.parametrize("value", [0, -1, float("inf"), -float("inf"), float("nan"), True, False])
def test_direct_timeout_rejects_nonpositive_or_nonfinite_values(value):
    with pytest.raises(ValueError, match="positive finite"):
        OllamaProvider(request_timeout=value)


@pytest.mark.parametrize("value", [0.1, 1, 45.5])
def test_direct_timeout_accepts_positive_finite_values(value):
    assert OllamaProvider(request_timeout=value).request_timeout == float(value)


@pytest.mark.parametrize("native", [
    {}, {"error": "model missing"}, {"message": None}, {"message": []},
    {"message": "text"}, [], None, {"message": {}},
    {"message": {"content": []}}, {"message": {"thinking": {}}},
    {"message": {"tool_calls": {}}},
    {"message": {"tool_calls": [None]}},
    {"message": {"tool_calls": [{"function": None}]}},
])
def test_malformed_completed_envelopes_are_classified_by_core(native):
    provider = RespondingOllamaProvider([native])
    with pytest.raises(ModelRequestError) as caught:
        ModelAdapter(provider, "model").respond([], [])
    details = model_request_error_details(caught.value)
    assert details["failure_class"] == "malformed_response"
    assert details["_transient"] is False
    assert type(caught.value.__cause__).__name__ == "_ResponseValidationError"


@pytest.mark.parametrize("status, category, transient", [
    (429, "rate_limit", True), (500, "provider_server_error", True),
    (503, "provider_server_error", True), (400, "configuration", False),
    (404, "configuration", False), (401, "authentication", False),
])
def test_interruptible_http_errors_preserve_core_classification(monkeypatch, status, category, transient):
    real_client = httpx.Client
    transport = httpx.MockTransport(lambda req: httpx.Response(
        status, json={"error": "fixture error"}, headers={"Retry-After": "7"}, request=req,
    ))
    monkeypatch.setattr("squid_providers.ollama.httpx.Client", lambda **kwargs:
        real_client(transport=transport, **kwargs))
    adapter = ModelAdapter(OllamaProvider(), "model")
    with pytest.raises(ModelRequestError) as caught:
        adapter.respond_streaming([], [], lambda _delta: None, should_cancel=lambda: False)
    wrapper = caught.value.__cause__
    assert type(wrapper) is RuntimeError
    assert isinstance(wrapper.__cause__, httpx.HTTPStatusError)
    for error in (wrapper, caught.value):
        details = model_request_error_details(error)
        assert details["http_status"] == status
        assert details["failure_class"] == category
        assert details["_transient"] is transient
        assert details["retry_after"] == "7"
        assert "fixture error" in details["response_body_excerpt"]
    assert model_request_error_details(wrapper)["underlying_error_type"] == "HTTPStatusError"


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("status, category, transient", [
    (429, "rate_limit", True), (500, "provider_server_error", True),
    (503, "provider_server_error", True), (400, "configuration", False),
    (404, "configuration", False), (401, "authentication", False),
])
def test_urllib_http_errors_preserve_core_classification(
    monkeypatch, streaming, status, category, transient,
):
    original = HTTPError(
        "http://localhost:11434/api/chat", status, "fixture HTTP error", {},
        BytesIO(b'{"error":"fixture error"}'),
    )

    def fail_request(*args, **kwargs):
        raise original

    monkeypatch.setattr("squid_providers.ollama.request.urlopen", fail_request)
    adapter = ModelAdapter(OllamaProvider(), "model")
    with pytest.raises(ModelRequestError) as caught:
        if streaming:
            # Omitting should_cancel selects the actual urllib streaming path.
            adapter.respond_streaming([], [], lambda _delta: None)
        else:
            adapter.respond([], [])

    wrapper = caught.value.__cause__
    assert type(wrapper) is RuntimeError
    assert wrapper.__cause__ is original
    # Check the provider wrapper itself as well as Core's additional wrapper.
    for error in (wrapper, caught.value):
        details = model_request_error_details(error)
        assert details["http_status"] == status
        assert details["failure_class"] == category
        assert details["_transient"] is transient
        # urllib's body is consumed by Ollama and retained in the safe message,
        # rather than exposed as httpx-style response_body_excerpt metadata.
        assert "fixture error" in details["error_message"]
        assert "response_body_excerpt" not in details
    assert model_request_error_details(wrapper)["underlying_error_type"] == "HTTPError"
    assert [item["error_type"] for item in
            model_request_error_details(caught.value)["exception_chain"]] == [
        "RuntimeError", "HTTPError",
    ]


@pytest.mark.parametrize("chunk", [
    None, [], {"error": "native failure", "done": True}, {"message": None, "done": True},
    {"message": {"content": []}}, {"message": {"tool_calls": [None]}},
    {"message": {"tool_calls": [{"function": []}]}},
])
def test_malformed_stream_envelopes_do_not_become_empty_success(chunk):
    provider = StreamingOllamaProvider([chunk])
    with pytest.raises(ModelRequestError) as caught:
        ModelAdapter(provider, "model").respond_streaming([], [], lambda _delta: None)
    assert model_request_error_details(caught.value)["failure_class"] == "malformed_response"
