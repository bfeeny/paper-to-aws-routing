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
  on unseen pairs, including cross-vendor and proprietary-vs-open pairs.
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

| Pair | Type |
| --- | --- |
| `claude-sonnet-5` / `claude-haiku-4-5` | same family, different size |
| `gpt-5.5` / `gpt-oss-20b` | same vendor, proprietary vs open weights |
| `qwen3-235b-a22b-2507` / `qwen3-32b` | open weights, different size |
| `claude-sonnet-5` / `qwen3-32b` | cross-vendor, proprietary vs open |

All are reachable through one `bedrock-mantle` connector target, which is what
makes the transfer test affordable.

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

_(none yet)_
