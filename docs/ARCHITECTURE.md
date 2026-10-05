# Squid Providers Architecture

Providers owns transports and an explicit first-party catalog. It imports only
squid_core.model_response and squid_core.providers.base, never Code Squid.
Transports return ModelResponse for streaming and non-streaming calls.
CancellationMonitor and raw tracing remain provider-owned; StreamCancelled
retains Core's single exception identity.

Application API in squid_providers.catalog: PROVIDER_PROFILES, PROFILES_BY_ID,
VALID_PROVIDERS, friendly_provider_name, resolve_provider_name,
validate_config, resolve_config, build_provider, provider_ready, discover_models,
and tool_call_content_mode. ProviderSettings and shared validators live in
squid_providers.settings. The catalog owns identities/defaults, credential
variable names, supported scoped config, constructors,
readiness/discovery and replay policy. request_timeout is supported by the native
transport profile; its default stays in the transport. Add providers explicitly
here; there is no plugin framework or dynamic entry-point system.

Code Squid -> Core; Code Squid -> Providers -> small Core contracts.
Each package lives in a sibling Git repository. Source is under the underscore
package, tests under tests/, packaging at the root. No compatibility source
paths remain. Applications persist sparse supported config dictionaries and
construct providers before injecting them into Core.

Provider image integration tests cover transport/replay behavior only.
Behavioral hook checks live in Code Squid alongside its hook implementations;
Providers has no test or production import dependency on application hooks.

Code Squid owns its initial/default provider choice. Providers supplies each
profile's connection defaults without choosing a backend for applications.
Only providers.<id> scoped connection configuration is supported; flat model
connection fields have no readers, migration helpers, or dual-write paths.
ProviderSettings and normally initialized transports are the current contracts;
missing constructor-owned fields are not tolerated for old objects or tests.

Ollama streaming appends complete native tool-call objects in arrival order,
including repeated tool names and supplied IDs. It does not merge calls by
chunk-local positions or guess at name/argument fragments. Native envelopes and
message field shapes are checked before normalization; validation errors remain
classifiable through Core, which alone parses and validates tool arguments.
Both direct and scoped Ollama timeouts must be positive and finite. Native URL
normalization still removes a trailing /v1 for interoperability.

LM Studio metadata requests send a configured bearer credential and remain
unauthenticated when none exists. Chat readiness requires a loaded LLM, using
the same type filter as discovery. Image capability behavior stays conservative.
Interruptible Ollama HTTP errors retain HTTPStatusError and its read response in
the exception chain, so Core classifies statuses and owns retry policy.
Both urllib Ollama request paths also chain HTTPError beneath RuntimeError.
Their consumed response bodies remain in the wrapper message, which Core bounds
and redacts; urllib failures do not supply a separate response_body_excerpt.
Regression tests exercise both transport families through Core's shared
model_request_error_details contract, including the adapter's additional wrapper.
Fallback OpenAI-compatible tool-call IDs have a response-specific prefix;
provider-supplied IDs remain unchanged in streaming and non-streaming replies.

Streamed usage is best-effort for v0.1: supplied counts are normalized, missing
counts remain unknown, and transports do not universally request stream_options
or retry incompatible requests to obtain usage. Cancellation identity and
transport ownership are unchanged; the two Ollama HTTP implementations remain.
