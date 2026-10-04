"""Provider-owned effective transport configuration, separate from runtime settings."""

from dataclasses import dataclass, field
from typing import Any


VALID_REASONING_MODES = frozenset({"off", "low", "medium", "high"})


@dataclass(frozen=True)
class ProviderSettings:
    """Effective model selection and provider-owned transport configuration."""

    provider: str
    model_name: str
    base_url: str | None
    api_key: str | None
    temperature: float | int | None = None
    reasoning_mode: str | None = None
    reasoning_token_budget: int | None = None
    config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.provider:
            raise ValueError("provider must not be empty")
        if not self.model_name:
            raise ValueError("model_name must not be empty")
        validate_temperature(self.temperature, "provider settings")
        validate_reasoning_mode(self.reasoning_mode, "provider settings")
        validate_reasoning_token_budget(self.reasoning_token_budget, "provider settings")
        # Import lazily to keep the catalog/configuration dependency acyclic.
        from .catalog import validate_config
        validate_config(self.provider, self.config)


def validate_temperature(value: Any, source: str = "provider settings") -> float | int | None:
    """Validate the shared sampling value for file and programmatic inputs."""

    if value is None:
        return None

    if isinstance(value, bool) or not isinstance(value, int | float) or not 0 <= value <= 2:
        raise ValueError(f"Invalid {source}: model.temperature must be null or a number from 0 through 2")

    return value


def validate_reasoning_mode(value: Any, source: str = "provider settings") -> str | None:
    """Validate normalized reasoning controls interpreted by transports."""

    if value is None:
        return None
    if not isinstance(value, str) or value not in VALID_REASONING_MODES:
        valid_modes = ", ".join(sorted(VALID_REASONING_MODES))
        raise ValueError(
            f"Invalid {source}: model.reasoning_mode must be null or one of {valid_modes}"
        )
    return value


def validate_reasoning_token_budget(value: Any, source: str = "provider settings") -> int | None:
    """Validate optional transport reasoning budgets, including zero."""

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            f"Invalid {source}: model.reasoning_token_budget must be null or a "
            "non-negative integer"
        )
    return value
