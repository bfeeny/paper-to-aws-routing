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

## Usage

```bash
make up ARM=always_strong        # deploy one arm
make smoke                       # one request end-to-end
make outputs                     # gateway URL and identifiers
make down ARM=always_strong      # remove it
make down-all                    # remove everything, including the artifact bucket
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

## Status

Scaffolding. No experimental runs yet; no results to cite.

## License

MIT — see [LICENSE](LICENSE).
