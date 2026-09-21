# Where each AgentCore capability fits this study

The study deliberately leans on Amazon Bedrock AgentCore where the platform earns it,
rather than everywhere it would technically work. This is the map, including the
capabilities we're consciously *not* using and why — a negative result about platform
fit is still a result.

Agent code, where any is needed, is written with **Strands Agents**. Strands is also
what AgentCore Evaluations expects to see in traces, so the choice removes an
impedance mismatch rather than adding one.

| Capability | Used? | Where it fits / why not |
| --- | --- | --- |
| **Gateway** | **Core** | The system under test. Inference targets front the models; a REQUEST interceptor rewrites `model` to implement routing. |
| **Evaluations** | **Yes — second opinion** | Managed pointwise scoring (`Builtin.Correctness`, `Builtin.Helpfulness`) alongside the repo's pairwise judge. Managed quota means scoring never contends with the experiment's own throughput. Not the headline metric: a managed rubric can change under us, so it corroborates rather than provides provenance. |
| **Observability** | **Yes** | The interceptor emits CloudWatch EMF per routing decision; OTel spans feed Evaluations. Every run is reconstructible from logs alone. |
| **Policy** | **Candidate arm** | Cedar policy could enforce per-team model allowlists at the gateway — a different way to constrain routing than an interceptor. Worth its own arm later: it adds an authorization hop, so the latency comparison is interesting. |
| **Identity** | **Not yet** | Callers authenticate with SigV4; there are no end users to attribute. Becomes relevant for a per-user budget or attribution experiment, where JWT claims drive routing. |
| **Runtime** | **If agents appear** | Single-turn prompts need no agent host. A multi-turn or tool-using variant would run Strands agents here rather than in Lambda. |
| **Memory** | **Not applicable** | Routing decisions here are stateless and single-turn. Directly relevant to the context-compaction study, which is a separate paper. |
| **Code Interpreter** | **No** | No code execution in the task set. Would matter for agentic benchmarks where answers must run. |
| **Browser** | **No** | No web interaction in the task set. |
| **Web Search** | **No** — deliberately | Live search injects non-determinism and makes runs unreproducible. A benchmark must be frozen. |
| **Optimization** (Insights / Recommendations / A-B tests) | **Evaluate later** | Native A/B testing overlaps with what the run harness does. Worth comparing once the harness produces stable numbers — if the managed version subsumes our arms, that's worth reporting. |
| **Agent Registry** | **No** | One experiment, no agent fleet to catalogue. |
| **Payments** | **No** | Out of scope. |

## Constraints found while integrating Evaluations

Recorded because they cost time and aren't obvious from the API reference:

1. **Spans must come from an allow-listed instrumentation scope.** `evaluate` rejects
   arbitrary OpenTelemetry spans with "no spans with supported scope". Supported
   scopes include `strands.telemetry.tracer`, `strands-agents`, the LangChain and
   LlamaIndex instrumentations, and OpenInference. The key in the span document is
   `scope`, not `instrumentation_scope`.
2. **Message content belongs in span events** (`gen_ai.user.message`, `gen_ai.choice`),
   not attributes — and as plain text. JSON-wrapped content leaks into the judge's
   reasoning, which we saw it comment on in an explanation.
3. **Evaluator level dictates the target.** TRACE-level evaluators (Correctness,
   Helpfulness) take `traceIds`; `spanIds` is only for TOOL_CALL-level evaluators.
4. **Ten evaluations per call.**

## Early methodological observation

On a 5-prompt validation slice, `Builtin.Correctness` scored **1.0 for every response
from both a 235B model and a 32B model** — no variance, no discrimination.
`Builtin.Helpfulness` varied per prompt but produced *identical means* for both arms.
The repo's pairwise judge separated the same two arms (25% win rate for the weak model)
and surfaced a 20% position-bias inconsistency rate.

The sample is far too small to conclude anything about the models. It is, however,
a clean illustration of why the primary metric is pairwise: **rubric scoring
saturates when comparing near-peer systems**, while forced preference still resolves
them. Both methods stay in the study precisely so this can be measured rather than
asserted.
