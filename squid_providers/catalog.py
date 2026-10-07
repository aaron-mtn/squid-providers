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
from .discovery import ModelRecord, discover_model_records

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
            request_timeout=settings.config.get("request_timeout"),
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
            reasoning_token_budget=settings.reasoning_token_budget,
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
    source: str = "Local"
    credential_fields: bool = True


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
        credential_fields=False,
    ),
    ProviderProfile(
        "openrouter",
        "OpenRouter",
        "OpenAI-compatible chat completions",
        "https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        source="Cloud",
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


def provider_catalog() -> list[dict[str, Any]]:
    """Return fresh, JSON-safe external provider metadata without credentials.

    Profiles remain the single source of defaults and policy. The existing
    managed transport profile is retained for runtime consumers, but applications
    compose their managed product entry separately from this external catalog.
    Never serialize a resolved config here: it can contain environment secrets.
    """
    return [
        {
            "id": profile.provider_id,
            "label": profile.friendly_name,
            "transport": profile.transport,
            "source": profile.source,
            "connection_fields": list(CONNECTION_FIELDS if profile.credential_fields else ("base_url",)),
            "config_fields": list(profile.config_fields),
            "defaults": {"base_url": profile.base_url, "api_key_env": profile.api_key_env or ""},
            "credentials_required": profile.credential_fields and not bool(profile.api_key),
        }
        for profile in PROVIDER_PROFILES if profile.provider_id != "squid_models"
    ]


def normalize_connection_config(
    provider_id: str,
    values: Mapping[str, Any] | None = None,
    previous: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge connection edits, retaining saved secrets on empty form submissions.

    Credential-free transports discard credential fields. Dummy transport keys
    are runtime defaults, never UI metadata. Other supported scoped options are
    preserved so connection edits cannot silently erase backend configuration.
    """
    profile = PROFILES_BY_ID[provider_id]
    result = {"base_url": profile.base_url, "api_key": profile.api_key or "",
              "api_key_env": profile.api_key_env or ""}
    result.update(validate_config(provider_id, previous or {}))
    edits = validate_config(provider_id, values or {})
    for name, value in edits.items():
        if name not in CONNECTION_FIELDS:
            result[name] = value
    if isinstance(edits.get("base_url"), str) and edits["base_url"].strip():
        result["base_url"] = edits["base_url"].strip()
    if edits.get("api_key"):
        result["api_key"] = edits["api_key"]
        result["api_key_env"] = ""
    elif edits.get("api_key_env"):
        result["api_key_env"] = edits["api_key_env"].strip()
        result["api_key"] = ""
    if not profile.credential_fields:
        result["api_key"] = result["api_key_env"] = ""
    elif result.get("api_key"):
        result["api_key_env"] = ""
    return result


def configuration_status(
    provider_id: str, overrides: Mapping[str, Any], *,
    environment: Mapping[str, str] | None = None,
) -> dict[str, bool]:
    """Report configuration availability without probing or returning secrets.

    Configuration is distinct from live readiness; provider_ready owns network
    checks. Missing credential variables are an ordinary unconfigured state.
    """
    profile = PROFILES_BY_ID[provider_id]
    effective = resolve_config(provider_id, overrides, environment=environment,
                               require_environment=False)
    available = bool(effective.get("api_key"))
    return {"credentials_configured": available,
            "configured": bool(effective.get("base_url")) and
                (not profile.credential_fields or available)}


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


def resolve_config(
    provider_id: str,
    overrides: Mapping[str, Any],
    *,
    environment: Mapping[str, str] | None = None,
    api_key_override: str | None = None,
    require_environment: bool = True,
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
        elif "api_key_env" in config and require_environment:
            raise ValueError(f"providers.{provider_id}.api_key_env names missing environment variable {variable!r}")
    return result


def provider_ready(settings: ProviderSettings) -> bool:
    """Observe inference readiness without propagating ordinary connection errors."""
    try:
        return build_provider(settings).is_ready()
    except (OSError, RuntimeError, ValueError):
        return False


def discover_models(settings: ProviderSettings) -> list[str]:
    """Return the normalized inventory's ordered IDs without losing exact case."""
    return [record.model_id for record in discover_model_records(settings)]
