"""Regression tests for transport-level model request cancellation."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import httpx
import pytest

from squid_core.providers.base import StreamCancelled
from squid_providers.ollama import OllamaProvider
from squid_providers.openai_compatible import OpenAICompatibleProvider


def _cancel_soon(cancelled: threading.Event, delay: float = 0.05) -> threading.Thread:
    thread = threading.Thread(
        target=lambda: (time.sleep(delay), cancelled.set()),
        name="test-cancel-provider",
    )
    thread.start()
    return thread


def _assert_no_cancellation_monitor() -> None:
    assert not any(
        thread.name == "squid-provider-cancellation" and thread.is_alive()
        for thread in threading.enumerate()
    )


class _BlockingOpenAIClient:
    def __init__(self, stream=None) -> None:
        self.closed = threading.Event()
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create),
        )
        self.stream = stream

    def _create(self, **_payload):
        if self.stream is not None:
            return self.stream
        assert self.closed.wait(2), "request client was not closed"
        raise RuntimeError("transport aborted")

    def close(self) -> None:
        self.closed.set()


class _BlockingOpenAIStream:
    def __init__(self, first_chunk=None) -> None:
        self.first_chunk = first_chunk
        self.sent_first = False
        self.closed = threading.Event()

    def __iter__(self):
        return self

    def __next__(self):
        if not self.sent_first and self.first_chunk is not None:
            self.sent_first = True
            return self.first_chunk
        assert self.closed.wait(2), "stream was not closed"
        raise RuntimeError("stream aborted")

    def close(self) -> None:
        self.closed.set()


def _openai_provider(client) -> OpenAICompatibleProvider:
    provider = OpenAICompatibleProvider.__new__(OpenAICompatibleProvider)
    provider.temperature = None
    provider.reasoning_mode = None
    provider.reasoning_token_budget = None
    provider.provider_name = "openai_compatible"
    provider.client = client
    return provider


def _chunk(*, content=None, reasoning_content=None):
    delta = SimpleNamespace(content=content, reasoning_content=reasoning_content)
    return SimpleNamespace(id="response", choices=[SimpleNamespace(delta=delta, finish_reason=None)])


def test_openai_cancellation_interrupts_wait_for_response_headers() -> None:
    cancelled = threading.Event()
    client = _BlockingOpenAIClient()
    timer = _cancel_soon(cancelled)
    started = time.monotonic()

    with pytest.raises(StreamCancelled):
        _openai_provider(client).stream_chat(
            "model", [], [], lambda _delta: None, should_cancel=cancelled.is_set
        )

    timer.join()
    assert time.monotonic() - started < 0.75
    assert client.closed.is_set()
    _assert_no_cancellation_monitor()


@pytest.mark.parametrize("reasoning_only", [False, True], ids=["text-gap", "reasoning-gap"])
def test_openai_cancellation_interrupts_long_chunk_gap(
    reasoning_only: bool,
) -> None:
    cancelled = threading.Event()
    first = _chunk(
        content=None if reasoning_only else "visible",
        reasoning_content="thought" if reasoning_only else None,
    )
    stream = _BlockingOpenAIStream(first)
    client = _BlockingOpenAIClient(stream)
    reasoning = []

    def cancel_after_delta(_delta: str) -> None:
        cancelled.set()

    with pytest.raises(StreamCancelled):
        _openai_provider(client).stream_chat(
            "model",
            [],
            [],
            cancel_after_delta if not reasoning_only else lambda _delta: None,
            on_reasoning_delta=cancel_after_delta if reasoning_only else None,
            should_cancel=cancelled.is_set,
        )

    assert stream.closed.is_set()
    assert client.closed.is_set()
    _assert_no_cancellation_monitor()


class _BlockingHTTPXResponse:
    def __init__(self, client, *, block_headers: bool) -> None:
        self.client = client
        self.block_headers = block_headers
        self.is_success = True
        self.status_code = 200
        self.closed = False

    def __enter__(self):
        if self.block_headers:
            assert self.client.closed.wait(2), "Ollama client was not closed"
            raise httpx.ReadError("transport aborted")
        return self

    def __exit__(self, *_args) -> None:
        self.closed = True

    def iter_lines(self):
        yield '{"message":{"content":"first"},"done":false}'
        assert self.client.closed.wait(2), "Ollama client was not closed"
        raise httpx.ReadError("transport aborted")

    def close(self) -> None:
        self.closed = True


class _BlockingHTTPXClient:
    instances = []
    block_headers = False

    def __init__(self, *, timeout) -> None:
        self.timeout = timeout
        self.closed = threading.Event()
        self.response = _BlockingHTTPXResponse(
            self, block_headers=type(self).block_headers
        )
        type(self).instances.append(self)

    def stream(self, method, url, *, json):
        return self.response

    def close(self) -> None:
        self.closed.set()


@pytest.mark.parametrize(
    "block_headers",
    [True, False],
    ids=["first-chunk", "read-gap"],
)
def test_ollama_cancellation_interrupts_active_transport(
    monkeypatch,
    block_headers,
) -> None:
    _BlockingHTTPXClient.instances = []
    _BlockingHTTPXClient.block_headers = block_headers
    monkeypatch.setattr("squid_providers.ollama.httpx.Client", _BlockingHTTPXClient)
    cancelled = threading.Event()
    timer = _cancel_soon(cancelled)
    started = time.monotonic()

    with pytest.raises(StreamCancelled):
        OllamaProvider(request_timeout=30).stream_chat(
            "model", [], [], lambda _delta: None, should_cancel=cancelled.is_set
        )

    timer.join()
    client = _BlockingHTTPXClient.instances[0]
    assert time.monotonic() - started < 0.75
    assert client.closed.is_set()
    if not block_headers:
        assert client.response.closed is True
    _assert_no_cancellation_monitor()
