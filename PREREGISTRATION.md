# Preregistration

Written before any experimental run. Committed first so the analysis plan is
timestamped ahead of the data. Deviations get appended to "Amendments" with a
date and a reason, rather than edited in silently.

**Status:** draft — not yet locked.
**Locked at commit:** _(fill in when the first run starts)_

## 1. Question

Do preference-trained LLM routers (RouteLLM, [arXiv:2406.18665](https://arxiv.org/abs/2406.18665))
retain their cost/quality advantage when deployed as an Amazon Bedrock AgentCore
Gateway REQUEST interceptor, and does the advantage transfer across model pairs
that differ in family, vendor, and open vs proprietary weights?

The paper reports >85% cost reduction on MT-Bench at no quality loss, and claims
routers transfer when the strong/weak pair is swapped. Both claims are tested here
on a platform the authors did not use, against the routing heuristic AWS itself
publishes.

## 2. Hypotheses

- **H1 (cost).** The trained router reaches the same judged quality as always-strong
  at materially lower cost than always-strong.
- **H2 (beats the heuristic).** At matched quality, the trained router costs less
  than the input-length heuristic from AWS's interceptor documentation.
- **H3 (transfer).** A router trained on one strong/weak pair retains its advantage
  on unseen pairs, including cross-vendor pairs and pairs with different parameter gaps.
- **H4 (latency).** Interceptor overhead is material relative to end-to-end latency,
  and large enough to change the deployment recommendation for interactive traffic.
- **H5 (accounting).** Savings measured in list-price dollars differ from savings
  measured in Bedrock quota burndown, because quota reserves `max_tokens` up front
  and weights newer Claude output tokens 5×.

H5 is the claim most likely to be novel: the paper's cost model assumes tokens are
counted once, at list price.

## 3. Arms

| Arm | Definition |
| --- | --- |
| `always_strong` | Every request to the strong model. Quality ceiling and cost ceiling. |
| `always_weak` | Every request to the weak model. Cost floor and quality floor. |
| `length` | Input-size threshold, reimplementing AWS's documented interceptor heuristic. |
| `routellm` | Trained router, decision threshold swept across the cost/quality curve. |

Each arm is a separate CloudFormation stack, so runs cannot contaminate each other.

## 4. Model pairs

Chosen from models this account can actually invoke (see §4a). All pairs are
open-weight, which also means a reproducer needs no gated model entitlements.

| Pair | Type |
| --- | --- |
| `qwen.qwen3-235b-a22b-2507` / `qwen.qwen3-32b` | same family, ~7× parameter gap |
| `openai.gpt-oss-120b` / `openai.gpt-oss-20b` | same family, ~6× gap |
| `mistral.mistral-large-3-675b-instruct` / `mistral.ministral-3-3b-instruct` | same vendor, extreme gap |
| `zai.glm-5` / `qwen.qwen3-32b` | cross-vendor |
| _(pending)_ `claude-sonnet` / `claude-haiku` | workhorse pair — the deployment most teams actually run |
| _(pending)_ `claude-opus` / `claude-haiku` | maximum price and capability gap |

Running several pair "gap sizes" (≈6×, ≈7×, extreme) tests something RouteLLM does not:
whether the router's advantage scales with the price ratio, and whether quality loss
scales with the capability gap.

`openai.gpt-oss-*` is the only family callable on both `/v1/chat/completions` and
`/v1/responses`, so it also serves as the control for API-shape effects.

### 4a. Catalog vs entitlement (measured 2026-09-20)

`/v1/models` lists **55** models; **38** are callable and **17** return
`permission_error: not available for this account`. The unavailable set is exactly
the gated commercial models: all Anthropic Claude, all OpenAI GPT-5.x, Gemma-4,
Grok. Raw probe: `results/model-availability/2026-09-20.jsonl`
(`runner/probe_models.py`).

**Anthropic models are doubly blocked in the study account (2026-09-20).** Beyond the
mantle entitlement, every Claude model in us-east-1 is inference-profile-only (0 of 13
support direct on-demand invocation), and both the `us.` and `global.` profiles route
to Regions denied by an organization SCP that restricts this account to us-east-1.
Claude pairs therefore require an SCP change in the management account, not just model
access. Pairs are config: adding them later is a config change plus an amendment here.

Two consequences for any router built on this endpoint:
1. **Model discovery cannot be trusted as a capability list.** A router that builds
   its catalog from `/v1/models` will route to models that 403 at call time.
2. **The API surface varies by family.** Claude is served from
   `/anthropic/v1/messages`, not `/v1/chat/completions`; `/v1/responses` is
   supported by only a subset. Cross-family routing therefore changes the request
   schema, not just the model string — a cost the paper's formulation does not model.

## 5. Data

- **Development set:** used for threshold sweeping and debugging.
- **Held-out set:** scored exactly once per arm, after thresholds are fixed.
  Reported numbers come only from the held-out set.
- Sets are fixed and committed before the first run; selection is by seeded shuffle,
  seed recorded in the run manifest.

## 6. Measures

**Quality.** Pairwise LLM-as-judge against the always-strong answer for the same
prompt. Each comparison is run in both orderings and a judgment counts only if it
is consistent across orderings; inconsistent pairs are reported as ties and their
rate is published. Judge model, prompt, and version are recorded per run.

**Cost.** Two independent ledgers per request:
1. list-price dollars from provider-reported token usage;
2. Bedrock quota burndown (input + reserved `max_tokens`, output weighted per model).

**Latency.** End-to-end gateway latency, and interceptor latency self-reported by the
Lambda via EMF. Overhead is the gateway path minus a direct `bedrock-mantle` call
on the same prompt.

## 7. Analysis

- Primary comparison: judged win-rate vs always-strong, plotted against cost per
  1,000 requests, per arm.
- **Every reported number carries a 95% bootstrap confidence interval** (10,000
  resamples over prompts). Differences whose intervals overlap are reported as
  inconclusive, not as wins.
- No arm is dropped after seeing results. Failed runs are reported with the failure.

## 8. Stopping rule

The held-out set is scored once per arm per pair. If a bug invalidates a run, the
run is discarded whole, the bug is recorded here, and the arm is re-run — partial
re-scoring is not permitted.

## 9. Known threats to validity

- Judge bias (position, verbosity, self-preference when the judge shares a family
  with a candidate). Mitigated by dual-ordering and by using a judge outside both
  families where possible.
- Provider-side variability: models behind an alias can change under us. Model IDs
  and run dates are recorded; results are valid for those IDs on those dates.
- Single region (us-east-1) and a small prompt budget; conclusions are about relative
  ordering of arms, not absolute throughput.
- Prompt sets are public benchmarks and may be contaminated in training data. This
  weakens absolute quality claims but affects all arms equally.

## Amendments

**2026-09-21 — baseline calibration on the full prompt set.** The first dev-split run
(qwen3-235b vs qwen3-32b, 20 prompts) decided only 8 of 20 pairs, with a win-rate CI of
0–75%: too under-powered to resolve anything, and the pair's 31% price gap caps what any
router could show. Baseline arms are therefore re-measured on all 80 prompts with a wider
pair (mistral-large-3-675b vs ministral-3-3b). No router and no threshold tuning is
involved, so nothing is fitted to these prompts; the final router comparison still
re-runs every arm, and held-out numbers remain the reported result. Judge changed to
`qwen.qwen3-235b-a22b-2507` because the previous judge (mistral-large-3) is now a
candidate and cannot grade itself.
