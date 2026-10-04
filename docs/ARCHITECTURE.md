# Squid Providers Architecture

Providers owns transports and an explicit first-party catalog. It imports only
squid_core.model_response and squid_core.providers.base, never Code Squid.
Transports return ModelResponse for streaming and non-streaming calls.
CancellationMonitor and raw tracing remain provider-owned; StreamCancelled
retains Core's single exception identity.

Application API in squid_providers.catalog: PROVIDER_PROFILES, PROFILES_BY_ID,
VALID_PROVIDERS, DEFAULT_PROVIDER_ID, friendly_provider_name, resolve_provider_name,
validate_config, resolve_config, build_provider, provider_ready, discover_models,
and tool_call_content_mode. ProviderSettings and shared validators live in
squid_providers.settings. The catalog owns identities/defaults, credential
variable names, supported scoped config, legacy interpretation, constructors,
readiness/discovery and replay policy. request_timeout is supported by the native
transport profile; its default stays in the transport. Add providers explicitly
here; there is no plugin framework or dynamic entry-point system.

Code Squid -> Core; Code Squid -> Providers -> small Core contracts.
Each package lives in a sibling Git repository. Source is under the underscore
package, tests under tests/, packaging at the root. No compatibility source
paths remain. Applications persist sparse supported config dictionaries and
construct providers before injecting them into Core.
