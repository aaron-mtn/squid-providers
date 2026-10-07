# Development log


- 2026-10-04: Finished the physical repository split and remaining provider
  catalog/config ownership after the earlier staged extraction. Initialized
  sibling squid-core and squid-providers Git repositories; moved runtime,
  transports, packaging and tests by ownership without keeping application-local
  source copies or compatibility re-exports. Git history and the earlier staged
  patch remain in Code Squid; no commits or remote publication were performed.
  ProviderProfiles, IDs/names/defaults, constructor choice, readiness/discovery,
  replay policy, credential variables, supported scoped config, shared transport
  control validation and legacy config interpretation now live in Providers.
  Code Squid consumes the catalog, persists arbitrary supported scoped overrides,
  applies its environment overrides and preserves model-selection UX. Removed
  application URL/key defaults and the named native timeout field; timeout is
  providers.<id>.request_timeout in provider-owned config. No concrete provider
  IDs/classes remain in production Code Squid or Core. Generic Core contracts,
  injected adapter/provider precedence, normalized responses, tracing ownership
  and cancellation semantics remain unchanged.
  Each repository now builds its own distribution. Core has generic reader
  dependencies only; Providers adds OpenAI/httpx; Code Squid depends on both
  local packages plus terminal dependencies. Requirements supply sibling editable
  paths together, avoiding lookup of unpublished packages. Updated README and
  architecture docs in all repositories; provider-extraction.md records moves.
  Split mixed agent/spawner/Mod/PDF/image tests; moved backend construction,
  transport/config/vision tests to Providers and generic runtime tests to Core.
  Retired old-path-only checks and added current import/source guards.
  Verification: focused Core (231), Providers (125) and corrected application
  settings/boundary/construction (120) checks passed during implementation.
  Final full suites in a fresh environment with sibling editable installs:
  Code Squid 706 passed; Core 847 passed, 1 platform skip; Providers 139 passed
  (1,692 passed total). Compilation and file-tree generation passed. All three
  wheels built and their contents/dependencies were inspected; they contain only
  their intended packages, without tests/bytecode. Installed all wheels together
  and imported them outside the checkout, constructed all catalog providers
  without requests, ran pip check and installed CLI --help. A separate Core-only
  wheel environment has no application, Providers, OpenAI or httpx and imported
  spawner/generation with those dependencies explicitly blocked. Fresh-process
  guards verify Providers imports only the two allowed Core modules with Code
  Squid blocked; source guards/searches verify no provider IDs/classes in Core
  or Code Squid. git diff --check passed in each repository; the complete
  application working-tree diff against HEAD also passed. The prior staged
  snapshot was left intact. No live inference was performed.
  Wheel inspection detected stale Core files in the old app build cache. Moved
  generated build/old distribution metadata to a temp archive and rebuilt cleanly;
  deleting those generated artifacts had been rejected by automatic review with
  the reason "blocked by policy". Deferred only commits, remote publication and
  package releases; no architecture or runtime work remains deferred.

- 2026-10-04: Removed behavioral-hook ownership from the mixed image
  integration test during sibling hook extraction. Kept missing replay asset
  transport coverage here, moved the TooManyReadsNudge image assertion to Code
  Squid and deleted the retired repeated-read assertion/import. No provider
  source, transport behavior or dependencies changed. Documented test ownership.
  Verification: focused image integration 4 passed; full Providers 139 passed.
  Cross-repo Core 734 passed, 1 skipped; Code Squid 777 passed. compileall and
  git diff --check passed. No live inference or deferred implementation work.

- 2026-10-05: Completed the confirmed pre-v0.1 provider cleanup. Native Ollama
  streaming now appends complete tool calls in arrival order instead of merging
  chunk-local positions/name/argument fragments. Added multi-chunk, repeated-name,
  multi-call, supplied-ID and ordering regressions. LM Studio native metadata
  preserves configured bearer authentication; readiness requires a loaded LLM.
  Interruptible Ollama HTTP failures retain HTTPStatusError and response metadata
  for Core classification. Completed/native stream envelopes and message field
  shapes now raise classifiable validation errors rather than empty successes;
  tool argument parsing/schema validation remains in Core. Direct timeout
  construction rejects nonpositive and non-finite values. OpenAI-compatible
  fallback call IDs have a response-specific prefix in both response paths.
  Deleted flat-settings migration exports and missing ProviderSettings field
  fallbacks. Normal provider construction replaces __new__ fixtures, and removed
  incomplete-instance branches/per-request client ownership compatibility.
  Code Squid owns default backend selection and consumes only scoped config;
  related application callers/tests/docs were updated in its sibling repository.
  Streamed usage remains best-effort: retain supplied counts, allow unknown
  usage, and do not universally request stream_options or add usage retries.
  Verification: focused Providers transport/catalog/cancellation 155 passed;
  full Providers 185 passed. Core adapter tests 63 passed; full Core 809 passed,
  1 skipped. Affected Code Squid settings/setup/runner/menu/CLI/boundary tests
  229 passed; full Code Squid 779 passed. Compilation of Providers source/tests,
  Code Squid source/tests and Core source passed. Constructed all five catalog
  providers without network requests; application imports, removed exports and
  shared cancellation identity passed. git diff --check passed in all three
  repositories. Core files and pre-existing work were left untouched. No live
  inference or release builds were performed. Ollama /v1 normalization, native
  reasoning aliases, conservative vision and cancellation design remain.
  Intentionally deferred broader vision, wrapper deduplication, test relocation,
  Ollama HTTP consolidation and package/version/license/release work. No further
  confirmed Fix before packaging issues were found within this patch.

- 2026-10-05: Verified finding #3 against Core's existing
  model_request_error_details classifier before further production edits. Core
  already follows explicit causes and unsuppressed contexts. Added 12 urllib
  HTTPError regressions across real non-streaming/streaming Ollama paths and six
  HTTP statuses; strengthened the six existing interruptible httpx regressions
  to classify the RuntimeError itself as well as ModelRequestError. Status,
  failure class and transient policy survive both chains. urllib body text stays
  in the safe wrapper message; httpx retains response_body_excerpt and retry_after.
  No production code or transport behavior changed for this verification.
  Verification: focused Ollama 88 passed; focused Core classifier/adapter 75
  passed; full Providers 197 passed; full Core 821 passed, 1 skipped. Compilation
  and git diff --check passed in both repositories. No live inference. Finding
  #3 is closed for these wrapping paths; no implementation follow-up deferred.

## 2026-10-05 - Provider-owned model discovery and ordering

- Added discovery.py and catalog ModelRecord/discover_model_records API; simple discover_models delegates. Moved native/compatible endpoints, parsing, fallback, display/load metadata, and normalization from Desktop. Exact IDs remain case-sensitive, duplicates merge useful metadata, and inventories sort by (ID.casefold(), ID). Raw transport order/readiness remains unchanged.
- Added fake-HTTP coverage for all catalog providers, metadata, fallbacks, empty/malformed responses, exact/case-distinct IDs, deduplication and raw transport order.
- Verification: full pytest suite 226 passed; compilation/import smoke checks and git diff --check passed. No live inference performed.


## 2026-10-06 - Safe external provider catalog for applications

- Added provider_catalog() as a JSON-safe projection of ProviderProfile, with external IDs/labels, transport/source, supported fields, non-secret URL/env-name defaults and credential requirements. Squid Models is excluded from this API; the existing managed transport profile remains unchanged for runtime consumers. No Desktop, Code Squid or Squid Models imports were added.
- Added provider-owned connection edit normalization and boolean configuration status. Empty form keys preserve saved secrets; credential-free transports clear credential fields; supported scoped backend options survive connection edits. resolve_config() keeps strict missing-variable behavior by default, with an explicit non-strict option for UI/preflight checks. Live readiness, inference and discovery semantics remain unchanged.
- Updated architecture docs and regressions for identities, separate LM Studio/compatible records, safe metadata, defaults, scoped fields, secret retention and environment-based configuration status. Code Squid profile consumers require no changes.
- Verification: full Providers suite 228 passed; compilation/import smoke checks and git diff --check passed. Desktop integration: 15 provider regressions and 41 focused route tests passed; all frontend suites passed. Its four full-suite failures also reproduce against untouched Desktop HEAD and concern existing Core/Voice integration issues.
