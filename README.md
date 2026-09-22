# paper-to-aws-routing

Reproducible study of **LLM request routing on Amazon Bedrock AgentCore Gateway**.

A router is deployed as a gateway REQUEST interceptor that rewrites the `model`
field before routing, and measured against two trivial baselines and the
input-length heuristic AWS publishes in its own interceptor documentation. Cost is
counted twice — list-price dollars and Bedrock quota burndown — because those are
not the same number.

The plan is fixed in advance: see [PREREGISTRATION.md](PREREGISTRATION.md).

## Why this setup

Every model in the study sits behind a single `bedrock-mantle` connector target, so
switching the strong/weak pair is a string change rather than a new integration.
That makes the *transfer* question — do routers survive swapping the model pair? —
cheap enough to answer across families, vendors and open vs proprietary weights.

## Requirements

- AWS account with Amazon Bedrock model access in `us-east-1`
- AWS CLI v2 (≥ 2.36) and Python 3.11+
- No other tooling: infrastructure is plain CloudFormation, driven by `make`

## Layout

```
infra/         CloudFormation: gateway, mantle target, interceptor, IAM
router/        interceptor handler and router implementations
experiments/   experiment definitions (arms, pairs, prompt sets, seeds)
runner/        executes a run; writes raw JSONL + a cost ledger
analysis/      scoring, bootstrap CIs, figures
results/       committed raw outputs, one directory per run
paper/         manuscript, figures built from results/
```

## Scoring

Two independent methods, deliberately:

```bash
python3 analysis/judge.py --baseline results/<strong-run> --candidate results/<arm-run>
python3 analysis/agentcore_eval.py --run results/<arm-run>
```

`judge.py` is the primary metric: pairwise, both orderings, prompt versioned here and
hashed into every summary. `agentcore_eval.py` is corroboration via **Amazon Bedrock
AgentCore Evaluations** — managed pointwise evaluators whose capacity doesn't consume
the experiment's own model quota. Where the two agree, a result is robust; where they
disagree, the disagreement is reported. See [docs/agentcore-fit.md](docs/agentcore-fit.md)
for which AgentCore capabilities this study uses, and which it deliberately doesn't.

## Tracking and teardown

Every AWS resource this study creates is named `routingstudy*` and tagged
`Project=paper-to-aws-routing`. To see what exists and what it would take to remove
it — including things CloudFormation doesn't own, like the packaging bucket and
orphaned log groups:

```bash
make inventory
```

`make down-all` removes every arm's stack plus the bucket. Bedrock model agreements
are account-level, cost nothing idle, and are deliberately *not* torn down by this
project.

## Usage

```bash
make up ARM=always_strong        # deploy one arm
make smoke                       # one request end-to-end
make outputs                     # gateway URL and identifiers
make down ARM=always_strong      # remove it
make down-all                    # remove everything, including the artifact bucket
make inventory                   # what exists right now, and how to delete it
make prices                      # refresh prices from the AWS Price List API
```

Scoring a run against a baseline:

```bash
python3 analysis/judge.py --baseline results/<strong-run> --candidate results/<arm-run>
```

Each arm is its own stack, so arms cannot contaminate each other, and an
interrupted study leaves nothing expensive running. `make down-all` is the cleanup
that matters: the gateway itself is billed per invocation, but a forgotten stack
is still an open door.

## Cost

The study is designed to run for tens of dollars: gateway invocations are billed per
1,000 calls, interceptor Lambdas are small and short, and model tokens dominate.
Lambda provisioned concurrency is deliberately **not** used — it would cost more per
month than the experiment, and the honest cold-start latency is itself a result.

## Early result: catalog vs entitlement

Before any routing experiment, `runner/probe_models.py` measured what the endpoint
will actually serve. On 2026-09-20, `/v1/models` listed **55** models of which
**38** were callable; the other **17** returned `not available for this account` —
all Anthropic Claude, all OpenAI GPT-5.x, Gemma-4 and Grok. The API surface also
varies by family: Claude is served from `/anthropic/v1/messages`, and `/v1/responses`
works for only a subset.

So a router cannot treat model discovery as a capability list, and cross-family
routing changes the request schema rather than just the model string. Raw data:
`results/model-availability/2026-09-20.jsonl`.

## Pricing

`make prices` fills `experiments/prices.json` from the **AWS Price List API**, keyed on
the exact usage type (`USE1-<model-id>-mantle-input-tokens-standard`), falling back to
the Marketplace offer rate card. No prices are transcribed by hand, and a model that
can't be resolved keeps `null` — the ledger still counts tokens but won't claim dollars
it can't source.

## Results so far

- **The method resolves a real capability gap.** `mistral-large-3-675b` vs
  `ministral-3-3b` over 80 prompts: weak win rate 12.8% (95% CI 2.9–24.3%), interval
  excluding 50%. A narrower pair (qwen3-235b vs qwen3-32b, 20 prompts) produced a CI of
  0–75% and resolved nothing — pair choice and sample size dominate everything else.
- **Position bias is large and grows with the gap.** 25% of pairs flipped when the
  answers were swapped on the narrow pair; **36%** on the wide one. Single-ordering
  judging silently absorbs that.
- **The trained router collapsed.** With 13 usable training labels it learned the base
  rate, scored every prompt within 0.075 of every other, and routed 100% to the strong
  model — while adding ~110 ms p50. See
  [results/reports/router-arm-2026-09-21.md](results/reports/router-arm-2026-09-21.md).
- **Quota burndown runs 1.28–1.37× actual tokens** on models with no output weighting,
  so cost measured in dollars and cost measured against quota are not the same number.

## Status

Harness complete and validated: infrastructure, run harness with dual cost ledger,
API-sourced prices, dual-ordering judge, AgentCore Evaluations as a second opinion, and
bootstrap intervals on every rate. The next step is real preference data — the current
router is label-starved, not mis-engineered.

## License

MIT — see [LICENSE](LICENSE).
