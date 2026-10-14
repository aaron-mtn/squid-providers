"""OpenAI-compatible chat-completions provider transport."""

import base64
import json
import time
import uuid
from typing import Any
from urllib import request
from urllib.error import HTTPError, URLError

from openai import OpenAI

from squid_core.model_response import ModelResponse, TokenUsage, ToolCall

from squid_core.providers.base import BaseProvider, StreamCancelled
from .cancellation import CancellationMonitor
from .discovery import cached_openrouter_capabilities, remember_openrouter_capabilities


class OpenAICompatibleProvider(BaseProvider):
    """Owns raw OpenAI-compatible API calls; normalizes completed output but must not own agent policy or tool execution."""

    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        temperature: float | int | None = None,
        reasoning_mode: str | None = None,
        reasoning_token_budget: int | None = None,
        provider_name: str = "openai_compatible",
    ) -> None:
        self.base_url = base_url
        # Keep the credential private to transport/setup requests. It must
        # never appear in status output or diagnostic representations.
        self._api_key = api_key
        self.temperature = temperature
        self.reasoning_mode = reasoning_mode
        self.reasoning_token_budget = reasoning_token_budget
        self.provider_name = provider_name
        self._image_capability_cache: dict[str, bool] = {}
        # OpenRouter's catalog is public, so discovery may construct this
        # transport without a credential. The inference client stays absent in
        # that case so a catalog request cannot accidentally weaken inference
        # authentication.
        self.client = (
            OpenAI(base_url=base_url, api_key=api_key)
            if api_key
            else None
        )

    def supports_streaming(self) -> bool:
        return True

    def supports_image_input(self, model: str) -> bool:
        """Use provider-advertised image input support; fail closed elsewhere."""

        if self.provider_name == "openrouter":
            capabilities = cached_openrouter_capabilities(self.base_url)
            if capabilities is None:
                # Manual model entry can bypass UI discovery. Reuse the same
                # public /models path, once per fresh catalog, rather than a
                # second modality endpoint or model-name heuristics.
                self.list_models()
                capabilities = cached_openrouter_capabilities(self.base_url)
            return capabilities is not None and capabilities.get(model, False)
        if self.provider_name != "lm_studio":
            return False
        cache = self._image_capability_cache
        if model in cache:
            return cache[model]
        supported = False
        for candidate in self._lm_studio_models():
            instance_ids = [
                instance.get("id")
                for instance in candidate.get("loaded_instances", [])
                if isinstance(instance, dict)
            ]
            if model != candidate.get("key") and model not in instance_ids:
                continue
            capabilities = candidate.get("capabilities")
            supported = (
                isinstance(capabilities, dict)
                and capabilities.get("vision") is True
            )
            break
        cache[model] = supported
        self._image_capability_cache = cache
        return supported

    def is_ready(self) -> bool:
        """Probe the backend using the most useful provider-native contract."""
        if self.provider_name == "openrouter" and not self._api_key:
            return False
        try:
            if self.provider_name == "squid_models":
                return self._get_json(self._service_root() + "/health") is not None
            if self.provider_name == "lm_studio":
                models = self._lm_studio_models()
                # A running LM Studio server without a loaded LLM cannot serve
                # the existing chat-completions inference path.
                return any(
                    model.get("type") == "llm" and model.get("loaded_instances")
                    for model in models
                )
            self.list_models()
            return True
        except (OSError, RuntimeError, ValueError, HTTPError, URLError):
            return False

    def list_models(self) -> list[str]:
        """Discover exact model IDs without changing the inference transport."""
        if self.provider_name == "lm_studio":
            models = self._lm_studio_models()
            return _unique_ids(
                model.get("key")
                for model in models
                if model.get("type") == "llm"
            )

        payload = self._get_json(
            self.base_url.rstrip("/") + "/models",
            authenticated=self.provider_name != "openrouter",
        )
        models = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(models, list):
            raise RuntimeError("Model discovery response did not include a model list")
        if self.provider_name == "openrouter":
            remember_openrouter_capabilities(self.base_url, models)
        return _unique_ids(
            model.get("id") for model in models if isinstance(model, dict)
        )

    def _lm_studio_models(self) -> list[dict]:
        payload = self._get_json(
            self._service_root() + "/api/v1/models",
        )
        models = payload.get("models") if isinstance(payload, dict) else None
        if not isinstance(models, list):
            raise RuntimeError("LM Studio response did not include a model list")
        return [model for model in models if isinstance(model, dict)]

    def _service_root(self) -> str:
        root = self.base_url.rstrip("/")
        return root[:-3].rstrip("/") if root.endswith("/v1") else root

    def _get_json(self, url: str, *, authenticated: bool = True) -> Any:
        headers = {"Accept": "application/json"}
        if authenticated and self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        http_request = request.Request(url, headers=headers, method="GET")
        with request.urlopen(http_request, timeout=0.5) as response:
            return json.loads(response.read().decode("utf-8"))

    def create_response(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict],
        tool_choice=None,
        please_stop_contemplating_the_void: bool = False,
        asset_loader=None,
    ) -> ModelResponse:
        payload = self._payload(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            please_stop_contemplating_the_void=please_stop_contemplating_the_void,
            asset_loader=asset_loader,
        )

        return adapt_response(self._inference_client().chat.completions.create(**payload))

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
        payload = self._payload(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            please_stop_contemplating_the_void=please_stop_contemplating_the_void,
            asset_loader=asset_loader,
        )
        payload["stream"] = True
        # Usage is best-effort: compatible backends may reject stream_options.
        # Normalize any supplied counts without universally requesting them.

        content_parts = []
        reasoning_parts = {"reasoning": [], "reasoning_content": [], "thinking": []}
        tool_call_parts = {}
        prepared_tool_call_indexes = set()
        request_started_at = time.perf_counter()
        state = {
            "usage": None,
            "finish_reason": None,
            "response_id": None,
            "raw_chunk_count": 0,
            "total_content_characters": 0,
            "total_reasoning_characters": 0,
            "first_raw_chunk_elapsed": None,
            "first_content_delta_elapsed": None,
            "first_reasoning_delta_elapsed": None,
            "saw_completion": False,
        }

        self._trace(
            trace_callback,
            {
                "type": "request_start",
                "timestamp_unix": time.time(),
                "monotonic": request_started_at,
                "provider": self.provider_name,
                "model": model,
                "base_url": self.base_url,
                "payload": _redacted_trace_payload(payload),
            },
        )

        client = self._streaming_request_client()
        stream = None

        def abort_request() -> None:
            # Before headers arrive only the client exists; afterwards closing
            # both layers also covers injected SDK-compatible stream doubles.
            close_stream = getattr(stream, "close", None)
            if callable(close_stream):
                close_stream()
            close_client = getattr(client, "close", None)
            if callable(close_client):
                close_client()

        monitor = CancellationMonitor(should_cancel, abort_request)
        try:
            with monitor:
                monitor.raise_if_cancelled()
                stream = client.chat.completions.create(**payload)
                self._consume_stream(
                    stream,
                    monitor,
                    trace_callback,
                    request_started_at,
                    content_parts,
                    reasoning_parts,
                    tool_call_parts,
                    prepared_tool_call_indexes,
                    on_text_delta,
                    on_reasoning_delta,
                    on_tool_preparing,
                    state,
                )
        except KeyboardInterrupt:
            raise
        except BaseException:
            monitor.raise_if_cancelled()
            raise
        finally:
            if stream is not None:
                close = getattr(stream, "close", None)
                if callable(close) and not monitor.cancelled:
                    close()
            if not monitor.cancelled:
                client.close()
            self._trace(
                trace_callback,
                {
                    "type": "request_complete",
                    "timestamp_unix": time.time(),
                    "elapsed": time.perf_counter() - request_started_at,
                    "response_id": state["response_id"],
                    "finish_reason": state["finish_reason"],
                    "first_raw_chunk_elapsed": state["first_raw_chunk_elapsed"],
                    "first_reasoning_delta_elapsed": state[
                        "first_reasoning_delta_elapsed"
                    ],
                    "first_visible_content_delta_elapsed": state[
                        "first_content_delta_elapsed"
                    ],
                    "total_reasoning_characters": state[
                        "total_reasoning_characters"
                    ],
                    "total_visible_content_characters": state[
                        "total_content_characters"
                    ],
                    "total_raw_chunk_count": state["raw_chunk_count"],
                },
            )

        if not state["saw_completion"]:
            if should_cancel is not None and should_cancel():
                raise StreamCancelled
            raise RuntimeError(
                "OpenAI-compatible stream ended before provider signaled completion"
            )

        return self._assembled_stream_response(
            content="".join(content_parts),
            tool_call_parts=tool_call_parts,
            usage=state["usage"],
            finish_reason=state["finish_reason"],
            response_id=state["response_id"],
            reasoning_parts=reasoning_parts,
        )

    def _consume_stream(
        self,
        stream,
        monitor,
        trace_callback,
        request_started_at,
        content_parts,
        reasoning_parts,
        tool_call_parts,
        prepared_tool_call_indexes,
        on_text_delta,
        on_reasoning_delta,
        on_tool_preparing,
        state,
    ) -> None:
        """Consume one SDK stream while its request client remains abortable."""
        for chunk in stream:
            monitor.raise_if_cancelled()
            chunk_elapsed = time.perf_counter() - request_started_at
            choices = getattr(chunk, "choices", []) or []
            if trace_callback is not None:
                self._trace_stream_chunk(
                    chunk,
                    choices,
                    chunk_elapsed,
                    state,
                    trace_callback,
                )

            state["response_id"] = state["response_id"] or getattr(
                chunk,
                "id",
                None,
            )
            chunk_usage = getattr(chunk, "usage", None)
            if chunk_usage is not None:
                state["usage"] = chunk_usage
            for choice in choices:
                choice_finish_reason = getattr(choice, "finish_reason", None)
                if choice_finish_reason is not None:
                    state["finish_reason"] = choice_finish_reason
                    # Transport EOF alone is not evidence of a complete reply.
                    state["saw_completion"] = True
                delta = getattr(choice, "delta", None)
                if delta is None:
                    continue
                for name, parts in reasoning_parts.items():
                    value = getattr(delta, name, None)
                    if value:
                        parts.append(value)
                if on_reasoning_delta is not None:
                    for name in reasoning_parts:
                        value = getattr(delta, name, None)
                        if value:
                            on_reasoning_delta(value)
                            break
                content_delta = getattr(delta, "content", None)
                if content_delta:
                    content_parts.append(content_delta)
                    on_text_delta(content_delta)
                for tool_call_delta in getattr(delta, "tool_calls", None) or []:
                    self._accumulate_tool_call_delta(tool_call_parts, tool_call_delta)
                    self._notify_tool_preparing(
                        tool_call_parts=tool_call_parts,
                        tool_call_delta=tool_call_delta,
                        prepared_tool_call_indexes=prepared_tool_call_indexes,
                        on_tool_preparing=on_tool_preparing,
                    )

    def _trace_stream_chunk(self, chunk, choices, elapsed, state, trace_callback) -> None:
        """Record raw streamed timing without changing response normalization."""
        state["raw_chunk_count"] += 1
        if state["first_raw_chunk_elapsed"] is None:
            state["first_raw_chunk_elapsed"] = elapsed
        content_deltas = []
        reasoning_deltas = {
            name: [] for name in ("reasoning", "reasoning_content", "thinking")
        }
        finish_reasons = []
        for choice in choices:
            finish_reason = getattr(choice, "finish_reason", None)
            if finish_reason is not None:
                finish_reasons.append(finish_reason)
            delta = getattr(choice, "delta", None)
            if delta is None:
                continue
            content = getattr(delta, "content", None)
            if content:
                content_deltas.append(content)
                state["total_content_characters"] += len(content)
                if state["first_content_delta_elapsed"] is None:
                    state["first_content_delta_elapsed"] = elapsed
            for name in reasoning_deltas:
                value = getattr(delta, name, None)
                if value:
                    reasoning_deltas[name].append(value)
                    state["total_reasoning_characters"] += len(value)
                    if state["first_reasoning_delta_elapsed"] is None:
                        state["first_reasoning_delta_elapsed"] = elapsed
        self._trace(
            trace_callback,
            {
                "type": "raw_chunk",
                "elapsed": elapsed,
                "chunk_index": state["raw_chunk_count"],
                "raw": chunk,
                "response_id": getattr(chunk, "id", None),
                "content_delta": "".join(content_deltas),
                "reasoning_delta": "".join(reasoning_deltas["reasoning"]),
                "reasoning_content_delta": "".join(
                    reasoning_deltas["reasoning_content"]
                ),
                "thinking_delta": "".join(reasoning_deltas["thinking"]),
                "finish_reason": finish_reasons,
            },
        )

    def _trace(self, trace_callback, event: dict) -> None:
        """Keep optional diagnostic output from changing provider behavior."""
        if trace_callback is None:
            return
        try:
            trace_callback(event)
        except Exception:
            pass

    def _payload(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict],
        tool_choice=None,
        please_stop_contemplating_the_void: bool = False,
        asset_loader=None,
    ) -> dict:
        payload = {
            "model": model,
            "messages": _messages_with_image_data(messages, asset_loader),
            "tools": tools,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if tools:
            payload["parallel_tool_calls"] = False
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        provider_name = self.provider_name
        reasoning_mode = self.reasoning_mode
        reasoning_token_budget = self.reasoning_token_budget
        if provider_name == "openrouter":
            # Omitted strictness can make upstream OpenAI routes require optional
            # fields. Default only on the wire; shared definitions and explicitly
            # configured strictness must survive unchanged across requests.
            payload["tools"] = [
                {**tool, "function": {"strict": False, **tool["function"]}}
                if tool.get("type") == "function" else tool
                for tool in tools
            ]
            if reasoning_token_budget is not None and reasoning_token_budget > 0:
                # OpenRouter documents max_tokens and effort as alternatives.
                # The explicit budget is more precise, so it wins when set.
                payload["extra_body"] = {
                    "reasoning": {"max_tokens": reasoning_token_budget},
                }
            elif reasoning_mode is not None:
                # OpenRouter calls disabled reasoning "none" on the wire;
                # Core's normalized user-facing value stays "off".
                effort = "none" if reasoning_mode == "off" else reasoning_mode
                payload["extra_body"] = {"reasoning": {"effort": effort}}
        if please_stop_contemplating_the_void:
            if provider_name == "squid_models":
                payload["extra_body"] = {
                    "chat_template_kwargs": {"enable_thinking": False},
                }
            elif provider_name == "openrouter":
                # This request-local override is authoritative even when the
                # provider has a normal numeric reasoning budget configured.
                payload["extra_body"] = {"reasoning": {"effort": "none"}}
        return payload

    def _accumulate_tool_call_delta(self, tool_call_parts: dict, tool_call_delta) -> None:
        index = getattr(tool_call_delta, "index", None)
        if index is None:
            index = len(tool_call_parts)

        part = tool_call_parts.setdefault(
            index,
            {
                "id": "",
                "type": "function",
                "name": "",
                "arguments": [],
            },
        )

        call_id = getattr(tool_call_delta, "id", None)
        if call_id:
            part["id"] = call_id

        call_type = getattr(tool_call_delta, "type", None)
        if call_type:
            part["type"] = call_type

        function = getattr(tool_call_delta, "function", None)
        if function is None:
            return

        name = getattr(function, "name", None)
        if name:
            part["name"] += name

        arguments = getattr(function, "arguments", None)
        if arguments:
            part["arguments"].append(arguments)

    def _notify_tool_preparing(
        self,
        tool_call_parts: dict,
        tool_call_delta,
        prepared_tool_call_indexes: set,
        on_tool_preparing,
    ) -> None:
        if on_tool_preparing is None:
            return

        index = getattr(tool_call_delta, "index", None)
        if index is None:
            index = len(tool_call_parts) - 1
        if index in prepared_tool_call_indexes:
            return

        part = tool_call_parts.get(index)
        if not part or not part["name"]:
            return

        prepared_tool_call_indexes.add(index)
        event = {
            "tool": part["name"],
            "tool_call_index": index,
        }
        if part["id"]:
            event["tool_call_id"] = part["id"]
        on_tool_preparing(event)

    def _assembled_stream_response(
        self,
        content: str,
        tool_call_parts: dict,
        usage,
        finish_reason=None,
        response_id=None,
        reasoning_parts=None,
    ) -> ModelResponse:
        # Native chunk assembly stays here; Core sees only ordered flat calls.
        # Missing native IDs still need to identify calls across transcript turns.
        fallback_id_prefix = f"openai_call_{uuid.uuid4().hex}"
        return ModelResponse(
            content=content,
            reasoning=_normalized_reasoning({
                name: "".join(parts) for name, parts in (reasoning_parts or {}).items()
            }),
            tool_calls=[ToolCall(
                id=part["id"] or f"{fallback_id_prefix}_{index}",
                name=part["name"],
                arguments="".join(part["arguments"]),
            ) for index, part in sorted(tool_call_parts.items())],
            usage=_adapt_usage(usage),
            finish_reason=finish_reason,
            response_id=response_id,
        )

    def _inference_client(self) -> OpenAI:
        """Return the authenticated client used only for model inference."""
        if self.client is None:
            raise ValueError(
                f"api_key is required for provider {self.provider_name!r} inference"
            )
        return self.client

    def _streaming_request_client(self) -> OpenAI:
        """Return an independently closable SDK client for one streamed call."""
        self._inference_client()
        return OpenAI(base_url=self.base_url, api_key=self._api_key)


def _unique_ids(values) -> list[str]:
    """Deduplicate non-empty provider IDs while preserving provider order."""
    seen: set[str] = set()
    model_ids: list[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        value = value.strip()
        if value and value not in seen:
            seen.add(value)
            model_ids.append(value)
    return model_ids


def _messages_with_image_data(messages: list[dict], asset_loader) -> list[dict]:
    """Materialize internal image references only at the transport boundary."""

    converted = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            converted.append(message)
            continue
        parts = []
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "image_asset":
                parts.append(block)
                continue
            if asset_loader is None:
                raise RuntimeError("session image asset loader is unavailable")
            data = asset_loader(block.get("asset_id"))
            encoded = base64.b64encode(data).decode("ascii")
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{block.get('mime_type')};base64,{encoded}",
                    },
                }
            )
        converted.append({**message, "content": parts})
    return converted


def _redacted_trace_payload(payload: dict) -> dict:
    """Keep trace diagnostics useful without retaining ephemeral image data."""

    messages = []
    for message in payload.get("messages", []):
        content = message.get("content")
        if not isinstance(content, list):
            messages.append(message)
            continue
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "image_url":
                parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": "[session image data]"},
                    }
                )
            else:
                parts.append(block)
        messages.append({**message, "content": parts})
    return {**payload, "messages": messages}


def _value(value, name):
    """Read SDK objects or native diagnostic fixtures at the provider boundary."""
    return value.get(name) if isinstance(value, dict) else getattr(value, name, None)


def _normalized_reasoning(message) -> str | None:
    # Completed output keeps the historical first nonempty alias precedence;
    # live deltas are independently handled by native stream assembly.
    for name in ("reasoning", "reasoning_content", "thinking"):
        value = _value(message, name)
        if value:
            return value
    return None


def _adapt_usage(usage) -> TokenUsage | None:
    if usage is None:
        return None

    def count(*names):
        for name in names:
            value = _value(usage, name)
            if value is not None:
                return value
        return None

    return TokenUsage(
        input_tokens=count("input_tokens", "prompt_tokens"),
        output_tokens=count("output_tokens", "completion_tokens"),
        total_tokens=count("total_tokens"),
    )


def adapt_response(response) -> ModelResponse:
    """Translate a native completed SDK response without parsing tool arguments."""
    try:
        choice = response.choices[0]
        message = choice.message
        fallback_id_prefix = f"openai_call_{uuid.uuid4().hex}"
        response_id = None
        for name in ("id", "response_id", "request_id", "_request_id"):
            response_id = _value(response, name)
            if response_id is not None:
                break
        return ModelResponse(
            content=message.content,
            reasoning=_normalized_reasoning(message),
            tool_calls=[ToolCall(
                _value(call, "id") or f"{fallback_id_prefix}_{index}",
                call.function.name,
                call.function.arguments,
            ) for index, call in enumerate(_value(message, "tool_calls") or [])],
            usage=_adapt_usage(_value(response, "usage")),
            finish_reason=_value(choice, "finish_reason"),
            response_id=response_id,
        )
    except (AttributeError, IndexError, KeyError, TypeError) as error:
        # Translation moved out of the adapter; keep native shape failures
        # classifiable as malformed responses with their original cause intact.
        raise _ResponseValidationError("Invalid completed provider response") from error


class _ResponseValidationError(ValueError):
    """Native completed output could not satisfy the Core response contract."""
