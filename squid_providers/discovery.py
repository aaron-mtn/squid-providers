"""Provider model inventory requests and normalization, coordinated by catalog.

This boundary owns discovery endpoints, parsing and fallbacks, not inference
readiness or application selection/presentation. Raw transport list_models()
methods retain their native ordering; public inventories are normalized here.
"""

from dataclasses import dataclass, replace
import json
import os
import time
from urllib import request
from urllib.error import HTTPError, URLError

from .settings import ProviderSettings
from .ollama import DEFAULT_OLLAMA_BASE_URL


MODEL_DISCOVERY_TIMEOUT_SECONDS = 10
OPENROUTER_CAPABILITY_TTL_SECONDS = 300
# Public catalog facts can be shared between UI discovery and a separately
# constructed inference transport. Scope them to the endpoint, never credentials,
# and expire them so changing provider metadata does not live forever.
_openrouter_image_capabilities: dict[str, tuple[float, dict[str, bool]]] = {}


def remember_openrouter_capabilities(base_url: str, models: list[dict]) -> None:
    """Retain authoritative input modalities before inventory drops raw rows."""
    capabilities = {}
    for model in models:
        if not isinstance(model, dict):
            continue
        model_id = model.get("id")
        if not isinstance(model_id, str) or not model_id.strip():
            continue
        architecture = model.get("architecture")
        modalities = architecture.get("input_modalities") if isinstance(architecture, dict) else None
        # Output image support, descriptions and model names are not evidence
        # of image *input*. Unknown or malformed metadata must fail closed.
        capabilities[model_id] = isinstance(modalities, list) and "image" in modalities
    _openrouter_image_capabilities[base_url.rstrip("/")] = (time.monotonic(), capabilities)


def cached_openrouter_capabilities(base_url: str) -> dict[str, bool] | None:
    """Return fresh catalog facts, or None when a catalog request is needed."""
    key = base_url.rstrip("/")
    cached = _openrouter_image_capabilities.get(key)
    if cached is None:
        return None
    observed_at, capabilities = cached
    if time.monotonic() - observed_at >= OPENROUTER_CAPABILITY_TTL_SECONDS:
        _openrouter_image_capabilities.pop(key, None)
        return None
    return capabilities.copy()


@dataclass(frozen=True)
class ModelRecord:
    """Exact provider ID with optional provider-advertised name and load state.

    None means the provider did not expose that metadata. Labels, selection,
    grouping and other application concepts intentionally do not belong here.
    """

    model_id: str
    display_name: str | None = None
    loaded: bool | None = None


def _text(value):
    if not isinstance(value, str):
        return None
    return value.strip() or None


def _normalize(records):
    """Deduplicate exact IDs, retaining useful metadata from duplicate rows."""
    unique = {}
    for record in records:
        model_id = record.model_id
        # IDs are opaque provider identifiers. Validate emptiness without
        # rewriting whitespace or case that may matter to a later API call.
        if not isinstance(model_id, str) or not model_id.strip():
            continue
        record = replace(record, model_id=model_id, display_name=_text(record.display_name))
        previous = unique.get(model_id)
        if previous is not None:
            # A duplicate loaded row must not disappear behind an unloaded row.
            record = ModelRecord(
                model_id,
                (record.display_name or previous.display_name)
                if record.loaded and not previous.loaded
                else (previous.display_name or record.display_name),
                True if previous.loaded or record.loaded else (
                    False if previous.loaded is False or record.loaded is False else None
                ),
            )
        unique[model_id] = record
    return sorted(unique.values(), key=lambda item: (item.model_id.casefold(), item.model_id))


def _root(base_url):
    root = _text(base_url)
    if root is None:
        raise ValueError("Provider base URL is not configured")
    return root.rstrip("/")


def _native_root(base_url):
    root = _root(base_url)
    return root[:-3].rstrip("/") if root.endswith("/v1") else root


def _get_json(url, api_key=None):
    """Bound discovery requests and keep errors compact and credential-free."""
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        with request.urlopen(request.Request(url, headers=headers),
                             timeout=MODEL_DISCOVERY_TIMEOUT_SECONDS) as response:
            payload = response.read()
    except HTTPError as exc:
        raise RuntimeError(f"Model discovery failed with HTTP {exc.code}") from exc
    except URLError as exc:
        raise RuntimeError(f"Model discovery request failed: {exc.reason}") from exc
    except TimeoutError as exc:
        raise RuntimeError("Model discovery timed out") from exc
    try:
        return json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Model discovery returned invalid JSON") from exc


def _rows(payload, field):
    rows = payload.get(field) if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise RuntimeError("Model discovery response did not include a model list")
    return [row for row in rows if isinstance(row, dict)]


def discover_model_records(settings: ProviderSettings, *, debug_log=None) -> list[ModelRecord]:
    """Return exact-ID unique inventory ordered by (ID.casefold(), ID).

    LM Studio and generic compatible services try native LLM metadata first,
    falling back to the compatible catalog when native discovery is unavailable
    or has no usable LLMs. A public OpenRouter catalog needs no inference key.
    """
    provider = settings.provider
    api_key = settings.api_key
    if not api_key and settings.config.get("api_key_env"):
        api_key = os.getenv(settings.config["api_key_env"])

    def log(message):
        if debug_log is not None:
            debug_log(f"model discovery: provider={provider} {message}")

    if provider == "ollama":
        url = _native_root(settings.base_url or DEFAULT_OLLAMA_BASE_URL) + "/api/tags"
        log(f"url={url}")
        return _normalize(ModelRecord(row.get("name") or row.get("model"))
                          for row in _rows(_get_json(url), "models"))
    if provider not in {"lm_studio", "openai_compatible", "openrouter", "squid_models"}:
        raise ValueError(f"Model discovery is not available for provider: {provider}")
    if provider in {"lm_studio", "openai_compatible"}:
        try:
            url = _native_root(settings.base_url) + "/api/v1/models"
            log(f"native_url={url}")
            records = _normalize(
                ModelRecord(row.get("key"), row.get("display_name"),
                            (isinstance(row.get("loaded_instances"), list)
                             and bool(row["loaded_instances"]))
                            if "loaded_instances" in row else None)
                for row in _rows(_get_json(url, api_key), "models")
                if row.get("type") == "llm"
            )
            if records:
                log(f"loaded_llms={sum(record.loaded is True for record in records)} total_llms={len(records)}")
                return records
            raise RuntimeError("LM Studio native response did not include usable LLM models")
        except (OSError, RuntimeError, ValueError) as exc:
            log(f"native_error={exc.__class__.__name__}: {exc}")
    url = _root(settings.base_url) + "/models"
    log(f"fallback_url={url}")
    models = _rows(_get_json(url, api_key), "data")
    if provider == "openrouter":
        remember_openrouter_capabilities(settings.base_url, models)
    return _normalize(ModelRecord(row.get("id"), row.get("name") or row.get("display_name"))
                      for row in models)
