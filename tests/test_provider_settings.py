"""Application provider configuration validation."""

import pytest

from squid_providers.settings import ProviderSettings


def test_provider_rejects_invalid_temperature_semantics() -> None:
    with pytest.raises(ValueError, match="temperature"):
        ProviderSettings(
            provider="openai_compatible",
            model_name="model",
            base_url=None,
            api_key=None,
            temperature=3,
        )


def test_provider_reasoning_mode_defaults_to_none() -> None:
    settings = ProviderSettings(
        provider="openai_compatible",
        model_name="model",
        base_url=None,
        api_key=None,
    )

    assert settings.reasoning_mode is None
    assert settings.reasoning_token_budget is None


@pytest.mark.parametrize("reasoning_token_budget", [0, 1, 4_000])
def test_provider_accepts_reasoning_token_budgets(
    reasoning_token_budget: int,
) -> None:
    settings = ProviderSettings(
        provider="openai_compatible",
        model_name="model",
        base_url=None,
        api_key=None,
        reasoning_token_budget=reasoning_token_budget,
    )

    assert settings.reasoning_token_budget == reasoning_token_budget


@pytest.mark.parametrize(
    "reasoning_token_budget",
    [True, False, -1, 1.5, "4000", [], {}],
)
def test_provider_rejects_invalid_reasoning_token_budgets(
    reasoning_token_budget: object,
) -> None:
    with pytest.raises(ValueError, match="reasoning_token_budget"):
        ProviderSettings(
            provider="openai_compatible",
            model_name="model",
            base_url=None,
            api_key=None,
            reasoning_token_budget=reasoning_token_budget,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("reasoning_mode", ["off", "low", "medium", "high"])
def test_provider_accepts_normalized_reasoning_modes(reasoning_mode: str) -> None:
    settings = ProviderSettings(
        provider="openai_compatible",
        model_name="model",
        base_url=None,
        api_key=None,
        reasoning_mode=reasoning_mode,
    )

    assert settings.reasoning_mode == reasoning_mode


@pytest.mark.parametrize(
    "reasoning_mode",
    ["auto", "max", "xhigh", "none", 1, ["high"]],
)
def test_provider_rejects_invalid_reasoning_modes(reasoning_mode: object) -> None:
    with pytest.raises(ValueError, match="reasoning_mode"):
        ProviderSettings(
            provider="openai_compatible",
            model_name="model",
            base_url=None,
            api_key=None,
            reasoning_mode=reasoning_mode,
        )
