"""Provider transport and generic completed-response regressions."""

from squid_core.model_response import ModelResponse, TokenUsage, ToolCall
from types import SimpleNamespace

import pytest

from squid_providers.openai_compatible import OpenAICompatibleProvider
from squid_core.providers.base import StreamCancelled


class FakeCompletions:
    def __init__(self) -> None:
        self.payloads = []
        self.response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="", tool_calls=[]))])

    def create(self, **payload):
        self.payloads.append(payload)
        return self.response


class FakeChat:
    def __init__(self) -> None:
        self.completions = FakeCompletions()


class FakeClient:
    def __init__(self) -> None:
        self.chat = FakeChat()


def provider_with_fake_client(
    temperature=None,
    reasoning_mode=None,
    reasoning_token_budget=None,
    provider_name: str = "openai_compatible",
) -> tuple[OpenAICompatibleProvider, FakeClient]:
    provider = OpenAICompatibleProvider.__new__(OpenAICompatibleProvider)
    provider.temperature = temperature
    provider.reasoning_mode = reasoning_mode
    provider.reasoning_token_budget = reasoning_token_budget
    provider.provider_name = provider_name
    provider.client = FakeClient()
    return provider, provider.client


def test_temperature_none_omits_temperature_from_payload() -> None:
    provider, client = provider_with_fake_client(temperature=None)

    provider.create_response(model="test-model", messages=[], tools=[])

    assert "temperature" not in client.chat.completions.payloads[0]


def test_zero_temperature_is_included_in_payload() -> None:
    provider, client = provider_with_fake_client(temperature=0)

    provider.create_response(model="test-model", messages=[], tools=[])

    assert client.chat.completions.payloads[0]["temperature"] == 0


def test_tools_present_disable_parallel_tool_calls() -> None:
    provider, client = provider_with_fake_client()

    provider.create_response(
        model="test-model",
        messages=[],
        tools=[{"type": "function", "function": {"name": "read_file"}}],
    )

    assert client.chat.completions.payloads[0]["parallel_tool_calls"] is False


def test_no_tools_omits_parallel_tool_calls() -> None:
    provider, client = provider_with_fake_client()

    provider.create_response(model="test-model", messages=[], tools=[])

    assert "parallel_tool_calls" not in client.chat.completions.payloads[0]


def test_tool_choice_is_forwarded_when_supplied() -> None:
    provider, client = provider_with_fake_client()

    provider.create_response(
        model="test-model",
        messages=[],
        tools=[],
        tool_choice={"type": "function", "function": {"name": "read_file"}},
    )

    assert client.chat.completions.payloads[0]["tool_choice"] == {
        "type": "function",
        "function": {"name": "read_file"},
    }


def test_generic_openai_compatible_omits_speculative_reasoning_fields() -> None:
    provider, client = provider_with_fake_client(provider_name="openai_compatible")

    provider.create_response(
        model="test-model",
        messages=[],
        tools=[],
        please_stop_contemplating_the_void=True,
    )

    payload = client.chat.completions.payloads[0]
    assert "extra_body" not in payload
    assert "reasoning" not in payload
    assert "chat_template_kwargs" not in payload


def test_managed_llama_cpp_disables_thinking_when_requested() -> None:
    provider, client = provider_with_fake_client(
        provider_name="squid_models",
        reasoning_token_budget=4_000,
    )

    provider.create_response(
        model="test-model",
        messages=[],
        tools=[],
        please_stop_contemplating_the_void=True,
    )

    assert client.chat.completions.payloads[0]["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False},
    }


def test_openrouter_disables_optional_reasoning_when_requested() -> None:
    provider, client = provider_with_fake_client(provider_name="openrouter")

    provider.create_response(
        model="test-model",
        messages=[],
        tools=[],
        please_stop_contemplating_the_void=True,
    )

    assert client.chat.completions.payloads[0]["extra_body"] == {
        "reasoning": {"effort": "none"},
    }


def test_reasoning_controls_are_omitted_by_default() -> None:
    for provider_name in ("squid_models", "openrouter"):
        provider, client = provider_with_fake_client(provider_name=provider_name)
        provider.create_response(model="test-model", messages=[], tools=[])
        assert "extra_body" not in client.chat.completions.payloads[0]


def test_openrouter_translates_normalized_reasoning_modes() -> None:
    expected_efforts = {
        "off": "none",
        "low": "low",
        "medium": "medium",
        "high": "high",
    }
    for reasoning_mode, effort in expected_efforts.items():
        provider, client = provider_with_fake_client(
            provider_name="openrouter",
            reasoning_mode=reasoning_mode,
        )

        provider.create_response(model="test-model", messages=[], tools=[])

        assert client.chat.completions.payloads[0]["extra_body"] == {
            "reasoning": {"effort": effort},
        }


def test_openrouter_uses_separate_reasoning_token_budget() -> None:
    provider, client = provider_with_fake_client(
        provider_name="openrouter",
        reasoning_token_budget=4_000,
    )

    provider.create_response(model="test-model", messages=[], tools=[])

    payload = client.chat.completions.payloads[0]
    assert payload["extra_body"] == {"reasoning": {"max_tokens": 4_000}}
    assert "max_tokens" not in payload
    assert "max_output_tokens" not in payload


def test_openrouter_explicit_budget_takes_precedence_over_reasoning_mode() -> None:
    provider, client = provider_with_fake_client(
        provider_name="openrouter",
        reasoning_mode="low",
        reasoning_token_budget=4_000,
    )

    provider.create_response(model="test-model", messages=[], tools=[])

    assert client.chat.completions.payloads[0]["extra_body"] == {
        "reasoning": {"max_tokens": 4_000},
    }


def test_openrouter_does_not_send_undocumented_zero_reasoning_budget() -> None:
    provider, client = provider_with_fake_client(
        provider_name="openrouter",
        reasoning_token_budget=0,
    )

    provider.create_response(model="test-model", messages=[], tools=[])

    assert "extra_body" not in client.chat.completions.payloads[0]


def test_openrouter_no_reasoning_override_takes_precedence_over_budget() -> None:
    provider, client = provider_with_fake_client(
        provider_name="openrouter",
        reasoning_mode="high",
        reasoning_token_budget=4_000,
    )

    provider.create_response(
        model="test-model",
        messages=[],
        tools=[],
        please_stop_contemplating_the_void=True,
    )

    assert client.chat.completions.payloads[0]["extra_body"] == {
        "reasoning": {"effort": "none"},
    }


def test_unsupported_providers_ignore_reasoning_token_budget() -> None:
    for provider_name in ("openai_compatible", "lm_studio", "squid_models"):
        provider, client = provider_with_fake_client(
            provider_name=provider_name,
            reasoning_token_budget=4_000,
        )

        provider.create_response(model="test-model", messages=[], tools=[])

        payload = client.chat.completions.payloads[0]
        assert "extra_body" not in payload
        assert "max_tokens" not in payload
        assert "max_output_tokens" not in payload


def test_generic_openai_compatible_ignores_normalized_reasoning_mode() -> None:
    provider, client = provider_with_fake_client(
        provider_name="openai_compatible",
        reasoning_mode="high",
    )

    provider.create_response(model="test-model", messages=[], tools=[])

    payload = client.chat.completions.payloads[0]
    assert "extra_body" not in payload
    assert "reasoning" not in payload


def test_transport_does_not_print_request_summary(capsys) -> None:
    provider, _client = provider_with_fake_client()

    provider.create_response(
        model="test-model",
        messages=[{"role": "user", "content": "secret task"}],
        tools=[{"type": "function", "function": {"name": "read_file"}}],
    )

    assert capsys.readouterr().out == ""


def chunk(content=None, tool_calls=None, *, reasoning_content=None, finish_reason=None, id=None):
    delta = SimpleNamespace(content=content)
    if tool_calls is not None:
        delta.tool_calls = tool_calls
    if reasoning_content is not None:
        delta.reasoning_content = reasoning_content
    return SimpleNamespace(
        id=id,
        choices=[SimpleNamespace(delta=delta, finish_reason=finish_reason)],
    )


def tool_call_delta(index, id=None, name=None, arguments=None):
    return SimpleNamespace(
        index=index,
        id=id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


class ClosableStream:
    """Small iterator double that records the provider cleanup contract."""

    def __init__(self, chunks) -> None:
        self.closed = False
        self.chunks = iter(chunks)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.chunks)

    def close(self) -> None:
        self.closed = True


def test_stream_chat_requests_stream_true_and_assembles_text() -> None:
    provider, client = provider_with_fake_client()
    client.chat.completions.response = [
        chunk(content="hello "),
        chunk(content="world", finish_reason="stop"),
    ]
    deltas = []

    response = provider.stream_chat(
        model="test-model",
        messages=[],
        tools=[],
        on_text_delta=deltas.append,
    )

    assert client.chat.completions.payloads[0]["stream"] is True
    assert deltas == ["hello ", "world"]
    assert response.content == "hello world"
    assert response.tool_calls == []


def test_stream_cancellation_stops_consumption_and_closes_response() -> None:
    provider, client = provider_with_fake_client()
    stream = ClosableStream([chunk(content="first"), chunk(content="second")])
    client.chat.completions.response = stream
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


def test_stream_chat_preserves_exposed_reasoning_finish_reason_and_response_id() -> None:
    provider, client = provider_with_fake_client()
    client.chat.completions.response = [
        chunk(reasoning_content="first ", id="chatcmpl_123"),
        chunk(reasoning_content="second", finish_reason="stop"),
    ]

    response = provider.stream_chat(
        model="test-model",
        messages=[],
        tools=[],
        on_text_delta=lambda _delta: None,
    )

    assert response.response_id == "chatcmpl_123"
    assert response.finish_reason == "stop"
    assert response.reasoning == "first second"
    assert isinstance(response, ModelResponse)


def test_stream_chat_forwards_normalized_reasoning_deltas() -> None:
    provider, client = provider_with_fake_client()
    client.chat.completions.response = [
        chunk(reasoning_content="first "),
        chunk(reasoning_content="second", finish_reason="stop"),
    ]
    reasoning_deltas = []

    provider.stream_chat(
        model="test-model",
        messages=[],
        tools=[],
        on_text_delta=lambda _delta: None,
        on_reasoning_delta=reasoning_deltas.append,
    )

    assert reasoning_deltas == ["first ", "second"]


def test_stream_chat_emits_reasoning_before_content_from_the_same_chunk() -> None:
    provider, client = provider_with_fake_client()
    client.chat.completions.response = [
        chunk(
            content="Visible.",
            reasoning_content="Last thought.",
            finish_reason="stop",
        )
    ]
    callbacks = []

    provider.stream_chat(
        model="test-model",
        messages=[],
        tools=[],
        on_text_delta=lambda delta: callbacks.append(("text", delta)),
        on_reasoning_delta=lambda delta: callbacks.append(("reasoning", delta)),
    )

    assert callbacks == [
        ("reasoning", "Last thought."),
        ("text", "Visible."),
    ]


def test_stream_trace_reports_raw_reasoning_before_visible_content() -> None:
    provider, client = provider_with_fake_client()
    client.chat.completions.response = [
        chunk(reasoning_content="thinking", id="chatcmpl_trace"),
        chunk(content="Ready.", finish_reason="stop"),
    ]
    events = []
    deltas = []

    provider.stream_chat(
        model="test-model",
        messages=[{"role": "user", "content": "prompt"}],
        tools=[],
        on_text_delta=deltas.append,
        trace_callback=events.append,
    )

    assert [event["type"] for event in events] == [
        "request_start", "raw_chunk", "raw_chunk", "request_complete",
    ]
    assert events[0]["payload"] == client.chat.completions.payloads[0]
    assert events[1]["raw"] is client.chat.completions.response[0]
    assert events[1]["reasoning_content_delta"] == "thinking"
    assert events[1]["content_delta"] == ""
    assert events[2]["content_delta"] == "Ready."
    assert deltas == ["Ready."]
    assert events[-1]["response_id"] == "chatcmpl_trace"
    assert events[-1]["first_raw_chunk_elapsed"] is not None
    assert events[-1]["first_reasoning_delta_elapsed"] is not None
    assert events[-1]["first_visible_content_delta_elapsed"] is not None
    assert events[-1]["total_reasoning_characters"] == len("thinking")
    assert events[-1]["total_visible_content_characters"] == len("Ready.")
    assert events[-1]["total_raw_chunk_count"] == 2


def test_stream_chat_assembles_tool_call_fragments() -> None:
    provider, client = provider_with_fake_client()
    client.chat.completions.response = [
        chunk(content="Reading file."),
        chunk(
            tool_calls=[
                tool_call_delta(
                    0,
                    id="call_1",
                    name="read_",
                    arguments='{"path": ',
                )
            ]
        ),
        chunk(
            tool_calls=[
                tool_call_delta(
                    0,
                    name="file",
                    arguments='"/tmp/a.txt"}',
                )
            ]
        ),
        chunk(finish_reason="tool_calls"),
    ]
    deltas = []

    response = provider.stream_chat(
        model="test-model",
        messages=[],
        tools=[{"type": "function", "function": {"name": "read_file"}}],
        on_text_delta=deltas.append,
    )

    tool_call = response.tool_calls[0]
    assert deltas == ["Reading file."]
    assert response.content == "Reading file."
    assert tool_call.id == "call_1"
    assert tool_call.name == "read_file"
    assert tool_call.arguments == '{"path": "/tmp/a.txt"}'


def test_stream_chat_keeps_interleaved_arguments_with_their_tool_call_indexes() -> None:
    provider, client = provider_with_fake_client(provider_name="openrouter")
    client.chat.completions.response = [
        chunk(
            tool_calls=[
                tool_call_delta(
                    0,
                    id="call_1",
                    name="write_file",
                    arguments='{"path":"/tmp/a.html",',
                ),
                tool_call_delta(
                    1,
                    id="call_2",
                    name="write_file",
                    arguments='{"path":"/tmp/b.css",',
                ),
            ]
        ),
        chunk(
            tool_calls=[
                tool_call_delta(1, arguments='"content":"body{}"}'),
                tool_call_delta(0, arguments='"content":"<main/>"}'),
            ]
        ),
        chunk(finish_reason="tool_calls"),
    ]

    response = provider.stream_chat(
        model="deepseek/deepseek-v4-flash-0731",
        messages=[],
        tools=[{"type": "function", "function": {"name": "write_file"}}],
        on_text_delta=lambda _delta: None,
    )

    assert [
        call.arguments for call in response.tool_calls
    ] == [
        '{"path":"/tmp/a.html","content":"<main/>"}',
        '{"path":"/tmp/b.css","content":"body{}"}',
    ]


def test_stream_chat_notifies_tool_preparing_once_when_name_appears() -> None:
    provider, client = provider_with_fake_client()
    client.chat.completions.response = [
        chunk(
            tool_calls=[
                tool_call_delta(
                    0,
                    id="call_1",
                    name="read_file",
                    arguments='{"path": ',
                )
            ]
        ),
        chunk(
            tool_calls=[
                tool_call_delta(
                    0,
                    arguments='"/tmp/a.txt"}',
                )
            ]
        ),
        chunk(finish_reason="tool_calls"),
    ]
    preparing_events = []

    provider.stream_chat(
        model="test-model",
        messages=[],
        tools=[{"type": "function", "function": {"name": "read_file"}}],
        on_text_delta=lambda _delta: None,
        on_tool_preparing=preparing_events.append,
    )

    assert preparing_events == [
        {
            "tool": "read_file",
            "tool_call_index": 0,
            "tool_call_id": "call_1",
        }
    ]


def test_stream_chat_tool_preparing_does_not_include_arguments() -> None:
    provider, client = provider_with_fake_client()
    client.chat.completions.response = [
        chunk(
            tool_calls=[
                tool_call_delta(
                    0,
                    id="call_1",
                    name="write_file",
                    arguments='{"path": "/tmp/a.txt", "content": "secret"}',
                )
            ]
        ),
        chunk(finish_reason="tool_calls"),
    ]
    preparing_events = []

    provider.stream_chat(
        model="test-model",
        messages=[],
        tools=[{"type": "function", "function": {"name": "write_file"}}],
        on_text_delta=lambda _delta: None,
        on_tool_preparing=preparing_events.append,
    )

    assert preparing_events == [
        {
            "tool": "write_file",
            "tool_call_index": 0,
            "tool_call_id": "call_1",
        }
    ]
    assert "arguments" not in preparing_events[0]
    assert "args" not in preparing_events[0]


@pytest.mark.parametrize(
    "chunks",
    [
        [chunk(content="partial")],
        [
            chunk(
                tool_calls=[
                    tool_call_delta(
                        0,
                        id="call_1",
                        name="read_file",
                        arguments='{"path": ',
                    )
                ]
            )
        ],
        [],
    ],
    ids=["partial-text", "partial-tool-call", "empty"],
)
def test_openai_compatible_incomplete_stream_fails_and_closes(chunks) -> None:
    provider, client = provider_with_fake_client()
    stream = ClosableStream(chunks)
    client.chat.completions.response = stream

    with pytest.raises(RuntimeError, match="ended before provider signaled completion"):
        provider.stream_chat(
            model="test-model",
            messages=[],
            tools=[],
            on_text_delta=lambda _delta: None,
        )

    assert stream.closed is True


def test_openai_compatible_completed_stream_closes() -> None:
    provider, client = provider_with_fake_client()
    stream = ClosableStream([chunk(content="complete", finish_reason="stop")])
    client.chat.completions.response = stream

    response = provider.stream_chat(
        model="test-model",
        messages=[],
        tools=[],
        on_text_delta=lambda _delta: None,
    )

    assert response.content == "complete"
    assert stream.closed is True


@pytest.mark.parametrize("aliases, expected", [
    ({"reasoning": "first", "reasoning_content": "second", "thinking": "third"}, "first"),
    ({"reasoning": "", "reasoning_content": "second", "thinking": "third"}, "second"),
    ({"thinking": "third"}, "third"),
    ({}, None),
])
def test_non_streaming_normalizes_native_completed_output(aliases, expected):
    provider, client = provider_with_fake_client()
    client.chat.completions.response = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=None, **aliases, tool_calls=[
                SimpleNamespace(id="a", extra="discard", function=SimpleNamespace(name="read_file", arguments='{"path":')),
                SimpleNamespace(id="b", function=SimpleNamespace(name="read_file", arguments='{}')),
            ]), finish_reason="tool_calls")],
        usage=SimpleNamespace(prompt_tokens=0, completion_tokens=3),
        id="response", request_id="ignored", extra="discard",
    )
    response = provider.create_response("test-model", [], [])
    assert isinstance(response, ModelResponse)
    assert response.content is None
    assert response.reasoning == expected
    assert response.usage == TokenUsage(0, 3, None)
    assert response.response_id == "response"
    assert response.finish_reason == "tool_calls"
    assert response.tool_calls == [ToolCall("a", "read_file", '{"path":'), ToolCall("b", "read_file", '{}')]
    assert not hasattr(response, "extra")
    assert not hasattr(response.tool_calls[0], "extra")


@pytest.mark.parametrize("id_name", ["response_id", "request_id", "_request_id"])
def test_native_response_id_aliases_are_provider_owned(id_name):
    provider, client = provider_with_fake_client()
    setattr(client.chat.completions.response, id_name, "native-id")
    assert provider.create_response("test-model", [], []).response_id == "native-id"


def test_stream_normalizes_usage_and_reasoning_precedence():
    first = chunk(reasoning_content="secondary", id="stream-id")
    first.choices[0].delta.reasoning = "primary"
    terminal = chunk(content="done", finish_reason="stop")
    terminal.usage = SimpleNamespace(prompt_tokens=5, completion_tokens=2, total_tokens=7)
    provider, client = provider_with_fake_client()
    client.chat.completions.response = [first, terminal]
    response = provider.stream_chat("test-model", [], [], lambda _delta: None)
    assert isinstance(response, ModelResponse)
    assert response.reasoning == "primary"
    assert response.usage == TokenUsage(5, 2, 7)
    assert response.response_id == "stream-id"


@pytest.mark.parametrize("streaming", [False, True])
def test_native_usage_keeps_missing_total_and_prefers_generic_aliases(streaming):
    provider, client = provider_with_fake_client()
    usage = SimpleNamespace(input_tokens=0, prompt_tokens=9, output_tokens=2, completion_tokens=8)
    if streaming:
        terminal = chunk(content="done", finish_reason="stop")
        terminal.usage = usage
        client.chat.completions.response = [terminal]
        response = provider.stream_chat("test-model", [], [], lambda _delta: None)
    else:
        client.chat.completions.response.usage = usage
        response = provider.create_response("test-model", [], [])
    assert response.usage == TokenUsage(0, 2, None)


def test_native_malformed_response_retains_failure_class_and_cause():
    from squid_core.model_adapter import ModelAdapter, ModelRequestError, model_request_error_details

    provider, client = provider_with_fake_client()
    client.chat.completions.response.choices = []
    with pytest.raises(ModelRequestError) as caught:
        ModelAdapter(provider, "test-model").respond([], [])
    assert model_request_error_details(caught.value)["failure_class"] == "malformed_response"
    assert isinstance(caught.value.__cause__.__cause__, IndexError)
