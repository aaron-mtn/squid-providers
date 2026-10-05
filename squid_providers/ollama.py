"""Native Ollama chat provider transport and response adapter."""

from __future__ import annotations

import base64
import json
import time
import uuid
from urllib import request
from urllib.error import HTTPError, URLError

import httpx

from squid_core.model_response import ModelResponse, TokenUsage, ToolCall

from squid_core.providers.base import BaseProvider, StreamCancelled
from .cancellation import CancellationMonitor


DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
# Local inference can legitimately take minutes, but it must still have a
# finite transport boundary so a stalled Ollama process cannot hang the agent.
DEFAULT_OLLAMA_REQUEST_TIMEOUT = 300.0


class OllamaProvider(BaseProvider):
    """Owns native Ollama /api/chat calls and translates completed output into Core ModelResponse."""

    def __init__(
        self,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        temperature: float | int | None = None,
        request_timeout: float | int | None = None,
    ) -> None:
        self.base_url = _native_base_url(base_url)
        self.temperature = temperature
        if request_timeout is None:
            request_timeout = DEFAULT_OLLAMA_REQUEST_TIMEOUT
        if (
            isinstance(request_timeout, bool)
            or not isinstance(request_timeout, int | float)
            or not 0 < request_timeout < float("inf")
        ):
            raise ValueError("Ollama request_timeout must be a positive finite number")
        self.request_timeout = float(request_timeout)
        self._image_capability_cache: dict[str, bool] = {}

    def supports_streaming(self) -> bool:
        return True

    def supports_image_input(self, model: str) -> bool:
        """Ask Ollama for the selected model's advertised capabilities."""

        cache = self._image_capability_cache
        if model in cache:
            return cache[model]
        body = json.dumps({"model": model, "verbose": False}).encode("utf-8")
        http_request = request.Request(
            f"{self.base_url}/api/show",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(http_request, timeout=self.request_timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        supported = "vision" in (payload.get("capabilities") or [])
        cache[model] = supported
        self._image_capability_cache = cache
        return supported

    def is_ready(self) -> bool:
        """Use Ollama's lightweight model-list request as its readiness probe."""
        try:
            self.list_models()
            return True
        except (OSError, RuntimeError, ValueError, HTTPError, URLError):
            return False

    def list_models(self) -> list[str]:
        """Return exact names from Ollama's native ``/api/tags`` API."""
        http_request = request.Request(
            f"{self.base_url}/api/tags",
            headers={"Accept": "application/json"},
            method="GET",
        )
        with request.urlopen(http_request, timeout=0.5) as response:
            payload = json.loads(response.read().decode("utf-8"))
        models = payload.get("models") if isinstance(payload, dict) else None
        if not isinstance(models, list):
            raise RuntimeError("Ollama response did not include a model list")
        seen: set[str] = set()
        model_ids: list[str] = []
        for model in models:
            if not isinstance(model, dict):
                continue
            model_id = model.get("name") or model.get("model")
            if isinstance(model_id, str) and model_id and model_id not in seen:
                seen.add(model_id)
                model_ids.append(model_id)
        return model_ids

    def create_response(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict],
        tool_choice=None,
        please_stop_contemplating_the_void: bool = False,
        asset_loader=None,
    ) -> ModelResponse:
        payload = {
            "model": model,
            "messages": convert_messages(messages, asset_loader=asset_loader),
            "tools": tools,
            "stream": False,
        }
        if please_stop_contemplating_the_void:
            payload["think"] = False
        if self.temperature is not None:
            payload["options"] = {"temperature": self.temperature}
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

        raw_response = self._post_chat(payload)
        return adapt_response(raw_response)

    def stream_chat(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict],
        on_text_delta,
        on_tool_preparing=None,
        tool_choice=None,
        trace_callback=None,
        please_stop_contemplating_the_void: bool = False,
        should_cancel=None,
        on_reasoning_delta=None,
        asset_loader=None,
    ) -> ModelResponse:
        payload = {
            "model": model,
            "messages": convert_messages(messages, asset_loader=asset_loader),
            "tools": tools,
            "stream": True,
        }
        if please_stop_contemplating_the_void:
            payload["think"] = False
        if self.temperature is not None:
            payload["options"] = {"temperature": self.temperature}
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

        assembled = _empty_stream_response()
        prepared_tool_call_indexes = set()
        request_started_at = time.perf_counter()
        raw_chunk_count = 0
        total_content_characters = 0
        total_reasoning_characters = 0
        first_raw_chunk_elapsed = None
        first_content_delta_elapsed = None
        first_reasoning_delta_elapsed = None
        response_id = None
        saw_completion = False
        self._trace(
            trace_callback,
            {
                "type": "request_start",
                "timestamp_unix": time.time(),
                "monotonic": request_started_at,
                "provider": "ollama",
                "model": model,
                "base_url": self.base_url,
                "payload": _redacted_trace_payload(payload),
            },
        )

        uses_builtin_stream = type(self)._post_chat_stream is OllamaProvider._post_chat_stream
        stream = (
            self._post_chat_stream_interruptible(payload)
            if should_cancel is not None and uses_builtin_stream
            else self._post_chat_stream(payload)
        )
        monitor = CancellationMonitor(should_cancel, stream.close)
        try:
            with monitor:
                monitor.raise_if_cancelled()
                for chunk in stream:
                    monitor.raise_if_cancelled()
                    if not isinstance(chunk, dict) or "error" in chunk:
                        raise _ResponseValidationError("Invalid Ollama stream envelope")
                    # A done-only terminal chunk is valid; an explicitly supplied
                    # message must still be an object before callbacks inspect it.
                    if "message" in chunk:
                        _validate_message_fields(chunk["message"])
                    chunk_elapsed = time.perf_counter() - request_started_at
                    raw_chunk_count += 1
                    if first_raw_chunk_elapsed is None:
                        first_raw_chunk_elapsed = chunk_elapsed
                    response_id = response_id or chunk.get("id")
                    message = chunk.get("message") or {}
                    content_delta = message.get("content") or ""
                    reasoning_deltas = {
                        name: message.get(name) or ""
                        for name in ("reasoning", "reasoning_content", "thinking")
                    }
                    if content_delta:
                        total_content_characters += len(content_delta)
                        if first_content_delta_elapsed is None:
                            first_content_delta_elapsed = chunk_elapsed
                    reasoning_character_count = sum(
                        len(value) for value in reasoning_deltas.values()
                    )
                    if reasoning_character_count:
                        total_reasoning_characters += reasoning_character_count
                        if first_reasoning_delta_elapsed is None:
                            first_reasoning_delta_elapsed = chunk_elapsed

                    self._trace(
                        trace_callback,
                        {
                            "type": "raw_chunk",
                            "elapsed": chunk_elapsed,
                            "chunk_index": raw_chunk_count,
                            "raw": chunk,
                            "response_id": chunk.get("id"),
                            "content_delta": content_delta,
                            "reasoning_delta": reasoning_deltas["reasoning"],
                            "reasoning_content_delta": reasoning_deltas["reasoning_content"],
                            "thinking_delta": reasoning_deltas["thinking"],
                            "finish_reason": chunk.get("done_reason"),
                        },
                    )
                    _accumulate_stream_chunk(
                        assembled,
                        chunk,
                        on_text_delta=on_text_delta,
                        on_reasoning_delta=on_reasoning_delta,
                        on_tool_preparing=on_tool_preparing,
                        prepared_tool_call_indexes=prepared_tool_call_indexes,
                    )
                    if chunk.get("done") is True:
                        saw_completion = True
                        break
            if not saw_completion:
                # EOF is not an Ollama completion signal. Check cancellation
                # once more because it may have closed an otherwise idle stream.
                if should_cancel is not None and should_cancel():
                    raise StreamCancelled
                raise RuntimeError(
                    "Ollama stream ended before provider signaled completion"
                )
        except KeyboardInterrupt:
            raise
        except BaseException:
            monitor.raise_if_cancelled()
            raise
        finally:
            close = getattr(stream, "close", None)
            if callable(close) and not monitor.cancelled:
                close()
            self._trace(
                trace_callback,
                {
                    "type": "request_complete",
                    "timestamp_unix": time.time(),
                    "elapsed": time.perf_counter() - request_started_at,
                    "response_id": response_id,
                    "finish_reason": assembled.get("done_reason"),
                    "first_raw_chunk_elapsed": first_raw_chunk_elapsed,
                    "first_reasoning_delta_elapsed": first_reasoning_delta_elapsed,
                    "first_visible_content_delta_elapsed": first_content_delta_elapsed,
                    "total_reasoning_characters": total_reasoning_characters,
                    "total_visible_content_characters": total_content_characters,
                    "total_raw_chunk_count": raw_chunk_count,
                },
            )

        return adapt_response(assembled)

    def _trace(self, trace_callback, event: dict) -> None:
        """Keep optional diagnostic output from changing provider behavior."""
        if trace_callback is None:
            return
        try:
            trace_callback(event)
        except Exception:
            pass

    def _post_chat(self, payload: dict) -> dict:
        body = json.dumps(payload).encode("utf-8")
        http_request = request.Request(
            f"{self.base_url}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(
                http_request,
                timeout=self.request_timeout,
            ) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            details = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama request failed with HTTP {error.code}: {details}") from error
        except URLError as error:
            # urllib nests socket timeouts in URLError.reason. Preserve that
            # concrete cause so the shared failure classifier sees a timeout.
            cause = error.reason if isinstance(error.reason, BaseException) else error
            raise RuntimeError(f"Ollama request failed: {error.reason}") from cause

    def _post_chat_stream(self, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        http_request = request.Request(
            f"{self.base_url}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(
                http_request,
                timeout=self.request_timeout,
            ) as response:
                for line in response:
                    line = line.strip()
                    if not line:
                        continue
                    yield json.loads(line.decode("utf-8"))
        except HTTPError as error:
            details = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama request failed with HTTP {error.code}: {details}") from error
        except URLError as error:
            cause = error.reason if isinstance(error.reason, BaseException) else error
            raise RuntimeError(f"Ollama request failed: {error.reason}") from cause

    def _post_chat_stream_interruptible(self, payload: dict):
        # httpx gives the provider an owned client that can be closed from the
        # cancellation monitor even while response headers or a body line are
        # blocked. urllib does not expose such a handle until urlopen returns.
        return _OllamaHTTPStream(
            f"{self.base_url}/api/chat",
            payload,
            timeout=self.request_timeout,
        )


class _OllamaHTTPStream:
    """One independently closable native Ollama streaming request."""

    def __init__(self, url: str, payload: dict, *, timeout: float) -> None:
        self._url = url
        self._payload = payload
        self._client = httpx.Client(timeout=timeout)
        self._response = None
        self._iterator = self._lines()

    def __iter__(self):
        return self

    def __next__(self) -> dict:
        return next(self._iterator)

    def close(self) -> None:
        if self._response is not None:
            self._response.close()
        self._client.close()

    def _lines(self):
        try:
            with self._client.stream("POST", self._url, json=self._payload) as response:
                self._response = response
                if not response.is_success:
                    # Read before raising so the retained response exposes its
                    # body and status to Core's generic failure classifier.
                    response.read()
                    response.raise_for_status()
                for line in response.iter_lines():
                    line = line.strip()
                    if line:
                        yield json.loads(line)
        except httpx.HTTPError as error:
            raise RuntimeError(f"Ollama request failed: {error}") from error


def _native_base_url(base_url: str | None) -> str:
    """Return an Ollama native base URL, tolerating existing OpenAI-compatible /v1 settings."""

    if not base_url:
        return DEFAULT_OLLAMA_BASE_URL

    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1"):
        return normalized[: -len("/v1")]
    return normalized


def convert_messages(messages: list[dict], asset_loader=None) -> list[dict]:
    converted = []
    tool_call_names: dict[str, str] = {}

    for message in messages:
        role = message.get("role")
        if role == "tool":
            converted.append(_convert_tool_result(message, tool_call_names))
            continue

        content, images = _ollama_content(message.get("content"), asset_loader)
        converted_message = {
            "role": role,
            "content": content,
        }
        if images:
            converted_message["images"] = images

        if role == "assistant" and message.get("tool_calls"):
            converted_tool_calls = []
            for tool_call in message["tool_calls"]:
                call_id = tool_call.get("id")
                function = tool_call.get("function", {})
                name = function.get("name")
                if call_id and name:
                    tool_call_names[call_id] = name

                converted_tool_calls.append(
                    {
                        "type": tool_call.get("type", "function"),
                        "function": {
                            "name": name,
                            "arguments": _parse_arguments(function.get("arguments", {})),
                        },
                    }
                )
            converted_message["tool_calls"] = converted_tool_calls

        converted.append(converted_message)

    return converted


def _ollama_content(content, asset_loader) -> tuple[str, list[str]]:
    """Convert provider-neutral blocks into Ollama text plus image arrays."""

    if not isinstance(content, list):
        return content or "", []
    text_parts = []
    images = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            text_parts.append(block["text"])
        elif block.get("type") == "image_asset":
            if asset_loader is None:
                raise RuntimeError("session image asset loader is unavailable")
            image_bytes = asset_loader(block.get("asset_id"))
            images.append(base64.b64encode(image_bytes).decode("ascii"))
    return "\n".join(text_parts), images


def _redacted_trace_payload(payload: dict) -> dict:
    messages = [
        (
            {**message, "images": ["[session image data]"] * len(message["images"])}
            if message.get("images")
            else message
        )
        for message in payload.get("messages", [])
    ]
    return {**payload, "messages": messages}


def _convert_tool_result(message: dict, tool_call_names: dict[str, str]) -> dict:
    tool_call_id = message.get("tool_call_id")
    tool_name = tool_call_names.get(tool_call_id)
    if not tool_name:
        raise ValueError(f"Ollama tool result references unknown tool_call_id: {tool_call_id}")

    return {
        "role": "tool",
        "tool_name": tool_name,
        "content": message.get("content") or "",
    }


def _parse_arguments(arguments):
    if not isinstance(arguments, str):
        return arguments

    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return arguments

    return parsed if isinstance(parsed, dict) else arguments


def adapt_response(raw_response: dict) -> ModelResponse:
    """Translate native output, preserving raw arguments for Core validation."""
    if not isinstance(raw_response, dict) or "error" in raw_response:
        raise _ResponseValidationError("Invalid completed Ollama response envelope")
    message = raw_response.get("message")
    _validate_message_fields(message)
    if not any(name in message for name in (
        "content", "tool_calls", "reasoning", "reasoning_content", "thinking",
    )):
        raise _ResponseValidationError("Ollama message must include completed output fields")
    native_calls = message.get("tool_calls")
    # Ollama may omit call IDs. One prefix per completed response keeps calls
    # readable by index without letting equal indexes in later turns collide.
    fallback_id_prefix = f"ollama_call_{uuid.uuid4().hex}"
    tool_calls = [
        _adapt_tool_call(index, tool_call, fallback_id_prefix)
        for index, tool_call in enumerate(native_calls or [])
    ]

    return ModelResponse(
        content=message.get("content") or "",
        reasoning=next((message.get(name)
                        for name in ("reasoning", "reasoning_content", "thinking")
                        if message.get(name)), None),
        tool_calls=tool_calls,
        usage=_adapt_usage(raw_response),
        finish_reason=raw_response.get("done_reason"),
        response_id=raw_response.get("id"),
    )


def _validate_message_fields(message) -> None:
    """Check wire structure without interpreting tool names or arguments."""
    if not isinstance(message, dict):
        raise _ResponseValidationError("Ollama response must include a message object")
    if message.get("content") is not None and not isinstance(message["content"], str):
        raise _ResponseValidationError("Ollama message content must be text or null")
    for name in ("reasoning", "reasoning_content", "thinking"):
        if message.get(name) is not None and not isinstance(message[name], str):
            raise _ResponseValidationError("Ollama message reasoning must be text or null")
    native_calls = message.get("tool_calls")
    if native_calls is not None and not isinstance(native_calls, list):
        raise _ResponseValidationError("Ollama message tool_calls must be a list or null")
    for call in native_calls or []:
        if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
            raise _ResponseValidationError("Ollama tool calls must include a function object")


class _ResponseValidationError(ValueError):
    """Native Ollama output cannot satisfy the completed response contract."""


def _adapt_tool_call(index: int, tool_call: dict, fallback_id_prefix: str) -> ToolCall:
    function = tool_call.get("function", {})
    arguments = function.get("arguments", {})
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments)

    return ToolCall(
        id=tool_call.get("id") or f"{fallback_id_prefix}_{index}",
        name=function.get("name"),
        arguments=arguments,
    )


def _empty_stream_response() -> dict:
    return {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [],
        },
    }


def _accumulate_stream_chunk(
    assembled: dict,
    chunk: dict,
    on_text_delta,
    on_reasoning_delta,
    on_tool_preparing,
    prepared_tool_call_indexes: set,
) -> None:
    message = chunk.get("message") or {}
    assembled_message = assembled["message"]

    for name in ("reasoning", "reasoning_content", "thinking"):
        value = message.get(name)
        if value:
            assembled_message[name] = f"{assembled_message.get(name, '')}{value}"
    if on_reasoning_delta is not None:
        for name in ("reasoning", "reasoning_content", "thinking"):
            value = message.get(name)
            if value:
                on_reasoning_delta(value)
                break

    content_delta = message.get("content")
    if content_delta:
        assembled_message["content"] += content_delta
        on_text_delta(content_delta)

    # Native Ollama emits complete calls, not OpenAI-style fragments. A
    # chunk-local position (or function index) must never merge separate calls.
    for tool_call in message.get("tool_calls") or []:
        index = len(assembled_message["tool_calls"])
        assembled_message["tool_calls"].append(tool_call)
        _notify_tool_preparing(
            assembled_message["tool_calls"],
            index,
            prepared_tool_call_indexes,
            on_tool_preparing,
        )

    for name in (
        "id",
        "done",
        "done_reason",
        "total_duration",
        "load_duration",
        "prompt_eval_count",
        "prompt_eval_duration",
        "eval_count",
        "eval_duration",
    ):
        if name in chunk:
            if name == "id":
                assembled[name] = assembled.get(name) or chunk[name]
            else:
                assembled[name] = chunk[name]


def _notify_tool_preparing(
    tool_calls: list[dict],
    index: int,
    prepared_tool_call_indexes: set,
    on_tool_preparing,
) -> None:
    if on_tool_preparing is None or index in prepared_tool_call_indexes:
        return

    function = tool_calls[index].get("function") or {}
    tool_name = function.get("name")
    if not tool_name:
        return

    prepared_tool_call_indexes.add(index)
    event = {
        "tool": tool_name,
        "tool_call_index": index,
    }
    call_id = tool_calls[index].get("id")
    if call_id:
        event["tool_call_id"] = call_id
    on_tool_preparing(event)


def _adapt_usage(raw_response: dict) -> TokenUsage | None:
    prompt_tokens = raw_response.get("prompt_eval_count")
    completion_tokens = raw_response.get("eval_count")
    if prompt_tokens is None and completion_tokens is None:
        return None

    total_tokens = None
    if prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens

    return TokenUsage(
        input_tokens=prompt_tokens,
        output_tokens=completion_tokens,
        total_tokens=total_tokens,
    )
