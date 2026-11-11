# Configurable guardrail transformations

AgentOPS previously used verdict-only guardrails. `PreLLMGuardrails.check` detects
input PII and blocks; `PostLLMGuardrails.check` detects response leakage and blocks.
The base agent reads `blocked`, `blocked_by` and `message`; it does not interpret
the post-check's existing `modifications` list as replacement response content.
Both guardrail classes expose async compatibility adapters. Local regex checks do
not need an LLM or additional inference tokens.

## Comparison with PicoAgents

| Behavior | Existing AgentOPS checks | PicoAgents PII middleware | New AgentOPS boundary middleware |
|---|---|---|---|
| Default PII response | Block | Replace matches | Block unless redaction is selected |
| Integration point | Outer request/response lifecycle | Model and tool operation hooks | Shared outer async lifecycle |
| Configuration | Enabled checks per pre/post stage | Middleware instances and regex replacements | Existing enabled checks plus per-check action |
| Input treatment | Extracted request text checked | String message content and shallow tool parameters transformed | Message text across current input and loaded history transformed |
| Output treatment | Extracted answer checked | Model message text and string tool results transformed | Assistant text and known Genie display fields transformed |
| Streaming protection | Post-check after delivery for native streams | Depends on operation hook placement and response shape | Post-redaction buffers final output before release |
| Tool/agent internals | Customer-owned | Intercepted when routed through PicoAgents hooks | Customer-owned; not automatically intercepted |

The PicoAgents implementation is an operation-level async-generator middleware
chain. Its PII redactor handles primarily string content and shallow tool fields;
it is not a complete nested-content or telemetry redaction solution. AgentOPS
uses its ResponsesAgent request/response contract and does not add PicoAgents as a
dependency or replace customer orchestration.

## Configuration

```yaml
guardrails:
  enabled: true
  pre_llm:
    enabled_checks: [pii, injection, input_length_max]
    pii:
      action: redact
      entities: [ssn, credit_card, email, phone]
  post_llm:
    enabled_checks: [pii_leakage, compliance, quality]
    pii_leakage:
      action: redact
      entities: [ssn, credit_card, email, phone]
```

Omitting `action` preserves blocking. A stage must select `pii` or `pii_leakage`
through `enabled_checks`, or use the existing default list. Global
`guardrails.enabled: false` bypasses transformations and checks. Other selected
checks still execute: redaction does not bypass injection or compliance checks.
Unknown actions or PII entities fail configuration validation. `entities` selects
what to redact; other entity types may still be blocked by the existing detector.

`PIIRedactionMiddleware.process_request` creates a new typed request. Input and
loaded history are transformed before checking and agent invocation; the original
request is not mutated. `process_response` creates a new typed response before
output checking and history persistence. IDs, roles, usage and authentication
fields remain intact. Genie table/chart display fields are transformed; arbitrary
custom output schemas require an explicit adapter. Existing documentation-domain
allowlists in the blocking output check are not applied by the strict redactor.

## Streaming

When post-LLM redaction is selected, the shared lifecycle buffers native output
and emits sanitized completed items after all post-checks. It suppresses raw
text deltas, so an email split across chunks cannot escape the redactor. This
trades token-by-token delivery for complete-response transformation. Other
post-check violations replace the buffered answer with the existing filtered
response. With redaction off, existing streaming behavior is retained.

## Trust and scope

A signed application context may authenticate the exact original messages.
Transforming them invalidates that signature. The generic middleware therefore
rejects input transformations when `custom_inputs.application_context` is present.
A team adapter must verify the original context and rebind its own authenticated
contract before this combination can be supported. The framework does not
silently bypass verification or rewrite a customer's signatures. Unchanged
messages continue through their normal verification path.

This feature protects the configured agent boundary. It does not automatically
transform customer-owned memory recall, nested model/tool calls, arbitrary
metadata, image contents, raw App storage or request/response traces captured
outside this boundary. Telemetry needs a separate tracing redaction policy.
Regex detection can miss PII and can produce false positives; production policies
may substitute a dedicated detector behind the same transformation boundary.

The supervisor's deployed configuration keeps blocking enabled. Redaction has
local integration coverage and is not enabled in the version-8 async rollout.
