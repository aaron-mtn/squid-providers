"""Explicit first-party provider catalog, defaults, configuration and construction.

This package owns backend policy. Applications persist sparse overrides and
choose when to probe services; Core receives a constructed provider only.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping

from squid_core.providers.base import BaseProvider
from .ollama import DEFAULT_OLLAMA_BASE_URL, OllamaProvider
from .openai_compatible import OpenAICompatibleProvider
from .settings import ProviderSettings

DEFAULT_PROVIDER_ID = "lm_studio"
CONNECTION_FIELDS = ("base_url", "api_key", "api_key_env")

OPENAI_COMPATIBLE_PROVIDERS = frozenset(
    {"lm_studio", "openai_compatible", "openrouter", "squid_models"}
)


def build_provider(
    settings: ProviderSettings,
    *,
    for_inference: bool = True,
) -> BaseProvider:
    """Construct the selected transport for inference or discovery."""

    # Selection belongs to the application; Core receives only the ready transport.

    if settings.provider == "ollama":
        return OllamaProvider(
            base_url=settings.base_url or DEFAULT_OLLAMA_BASE_URL,
            temperature=settings.temperature,
            request_timeout=getattr(settings, "config", {}).get("request_timeout"),
        )

    if settings.provider in OPENAI_COMPATIBLE_PROVIDERS:
        if settings.base_url is None:
            raise ValueError(f"base_url is required for provider {settings.provider!r}")
        if not settings.api_key and (for_inference or settings.provider != "openrouter"):
            raise ValueError(f"api_key is required for provider {settings.provider!r}")
        return OpenAICompatibleProvider(
            base_url=settings.base_url,
            api_key=settings.api_key,
            temperature=settings.temperature,
            reasoning_mode=settings.reasoning_mode,
            reasoning_token_budget=getattr(settings, "reasoning_token_budget", None),
            provider_name=settings.provider,
        )

    raise ValueError(f"Unsupported built-in provider: {settings.provider}")


def tool_call_content_mode(settings: ProviderSettings) -> str:
    """Return replay content behavior intrinsic to the selected backend."""

    # OpenRouter requires null content for assistant tool-call replay. Keep
    # backend policy here; ModelAdapter only implements the selected mode.
    return "null" if settings.provider == "openrouter" else "empty"


@dataclass(frozen=True)
class ProviderProfile:
    """Provider-owned defaults and supported configuration for one backend."""

    provider_id: str
    friendly_name: str
    transport: str
    base_url: str
    api_key: str | None = None
    api_key_env: str | None = None
    config_fields: tuple[str, ...] = ()


PROVIDER_PROFILES = (
    ProviderProfile(
        "squid_models",
        "Squid Models",
        "OpenAI-compatible chat completions",
        "http://127.0.0.1:8080/v1",
        api_key="local",
    ),
    ProviderProfile(
        "lm_studio",
        "LM Studio",
        "OpenAI-compatible chat completions",
        "http://127.0.0.1:1234/v1",
        api_key="lm-studio",
    ),
    ProviderProfile(
        "ollama",
        "Ollama",
        "native Ollama chat",
        "http://127.0.0.1:11434",
        config_fields=("request_timeout",),
    ),
    ProviderProfile(
        "openrouter",
        "OpenRouter",
        "OpenAI-compatible chat completions",
        "https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
    ),
    ProviderProfile(
        "openai_compatible",
        "OpenAI Compatible",
        "OpenAI-compatible chat completions",
        "http://127.0.0.1:8000/v1",
        api_key="local",
    ),
)
PROFILES_BY_ID = {profile.provider_id: profile for profile in PROVIDER_PROFILES}
VALID_PROVIDERS = frozenset(PROFILES_BY_ID)


def friendly_provider_name(provider_id: str) -> str:
    """Return the stable product label for a provider identity."""
    profile = PROFILES_BY_ID.get(provider_id)
    return profile.friendly_name if profile is not None else provider_id


def resolve_provider_name(value: str) -> str | None:
    """Resolve friendly names and convenient ID spellings case-insensitively."""
    normalized = " ".join(value.strip().lower().replace("-", "_").split())
    for profile in PROVIDER_PROFILES:
        choices = {
            profile.provider_id.lower(),
            profile.provider_id.replace("_", " ").lower(),
            profile.friendly_name.lower(),
        }
        if normalized in choices:
            return profile.provider_id
    return None


def validate_config(provider_id: str, overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Validate supported scoped fields without leaking their names into apps."""
    if provider_id not in PROFILES_BY_ID:
        raise ValueError(f"Unsupported built-in provider: {provider_id}")
    profile = PROFILES_BY_ID[provider_id]
    allowed = set(CONNECTION_FIELDS) | set(profile.config_fields)
    unknown = set(overrides) - allowed
    if unknown:
        raise ValueError(f"providers.{provider_id}: unsupported configuration: {', '.join(sorted(unknown))}")
    for name in CONNECTION_FIELDS:
        value = overrides.get(name)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"providers.{provider_id}.{name} must be a string or null")
    timeout = overrides.get("request_timeout")
    if timeout is not None and (isinstance(timeout, bool) or not isinstance(timeout, (float, int)) or not 0 < timeout < float("inf")):
        raise ValueError(f"providers.{provider_id}.request_timeout must be null or a positive finite number")
    return dict(overrides)


def legacy_model_config(provider_id: str, model: Mapping[str, Any]) -> dict[str, Any]:
    """Interpret the provisional flat format; scoped values remain authoritative."""
    result = {name: model[name] for name in CONNECTION_FIELDS if name in model}
    if provider_id == "ollama" and "ollama_request_timeout" in model:
        result["request_timeout"] = model["ollama_request_timeout"]
    return result


def remove_legacy_model_config(model: dict[str, Any]) -> None:
    """Remove flat fields after a writer has transferred them to scoped config."""
    for name in (*CONNECTION_FIELDS, "ollama_request_timeout"):
        model.pop(name, None)


def resolve_config(
    provider_id: str,
    overrides: Mapping[str, Any],
    *,
    environment: Mapping[str, str] | None = None,
    api_key_override: str | None = None,
) -> dict[str, Any]:
    """Apply provider defaults and credential variables to sparse saved overrides.

    Applications pass their product-wide environment overrides as scoped values.
    Provider credential environment names and missing-variable rules live here.
    """
    env = os.environ if environment is None else environment
    profile = PROFILES_BY_ID[provider_id]
    config = validate_config(provider_id, overrides)
    result = {"base_url": profile.base_url, "api_key": profile.api_key, "api_key_env": profile.api_key_env} | config
    variable = result["api_key_env"]
    if api_key_override is not None:
        result["api_key"] = api_key_override
    elif variable is not None:
        secret = env.get(variable)
        if secret is not None:
            result["api_key"] = secret
        elif "api_key_env" in config:
            raise ValueError(f"providers.{provider_id}.api_key_env names missing environment variable {variable!r}")
    return result


def provider_ready(settings: ProviderSettings) -> bool:
    """Observe inference readiness without propagating ordinary connection errors."""
    try:
        return build_provider(settings).is_ready()
    except (OSError, RuntimeError, ValueError):
        return False


def discover_models(settings: ProviderSettings) -> list[str]:
    """Discover sorted unique IDs, including credential-free public catalogs."""
    return sorted(set(build_provider(settings, for_inference=False).list_models()))
