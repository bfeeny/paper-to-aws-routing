# Router arm: a degenerate scorer, and why

**Run:** `20260921T040848Z-mistral-gap-mistral-wide-routellm-heldout` (60 held-out prompts)
**Pair:** `mistral-large-3-675b` (strong) vs `ministral-3-3b` (weak)
**Router:** Titan v2 embeddings (256-d) → L2 logistic regression, trained on judged pairs

## What happened

The router routed **60 of 60 requests to the strong model**. Cost and quality are
therefore identical to the `always_strong` arm, plus the router's own overhead.
It did not fail or fall back: every decision was logged as `scored`.

## Why: the scores do not separate prompts

| | value |
| --- | --- |
| min | 0.727 |
| p25 | 0.761 |
| median | 0.772 |
| p75 | 0.780 |
| max | 0.802 |

Every prompt scores within **0.075** of every other. The model has learned the base
rate — "most prompts need the strong model" — and almost nothing about *which* ones.

That collapse destroys the mechanism the paper depends on. A router is supposed to
expose a cost/quality curve you can tune; sweeping the threshold here gives a cliff:

| threshold | routed to weak |
| --- | --- |
| 0.50 | 0% |
| 0.70 | 0% |
| 0.75 | 7% |
| 0.78 | **75%** |
| 0.80 | 98% |
| 0.85 | 100% |

Between 0.75 and 0.80 the arm flips from all-strong to all-weak. There is no operating
point in between, so there is no curve to trade along — only a switch.

## The cause: 13 training labels

Labels came from our own judged pairs (`baseline_win` → strong needed; win or tie →
weak sufficed; inconsistent verdicts dropped). Of 80 prompts:

- 29 were dropped because **the judge contradicted itself** when the answers were swapped
- 51 usable labels, of which only **13 fell in the dev split** available for training
- held-out accuracy **0.632**, exactly equal to the **majority-class baseline 0.632**

RouteLLM trained on roughly 80,000 Chatbot Arena battles. Three orders of magnitude
separate that from this, and the gap shows up precisely where it should: in the
variance of the score, not in the plumbing.

**Bootstrapping labels from your own judge does not escape the problem.** The judge's
36% inconsistency rate is what destroyed a third of the training set — the same
weakness that limits the evaluation also starves the model that evaluation was meant
to train.

## What the router cost

| | value |
| --- | --- |
| router latency, warm p50 | **109 ms** |
| router latency, warm p90 | **128 ms** |
| router latency, cold start | 1,344 ms |
| savings delivered | **0%** |

The embedding hop is the price of a learned router on this platform: roughly 110 ms
added to every request, against AWS's reported ~93 ms p50 for an interceptor that only
rewrites a field. A router must earn that back in model spend before it is worth
deploying — and this one earned nothing.

## What would fix it

1. **Real preference data at scale.** Train on Chatbot Arena battles, as the paper did,
   rather than on labels bootstrapped from a 60-prompt run.
2. **Better labels per prompt.** Multiple judges or repeated sampling would rescue some
   of the 29 discarded pairs, at proportional cost.
3. **A cheaper feature than an embedding call** if the latency budget is tight — the
   `length` heuristic costs nothing and is the honest baseline to beat.

Item 1 is the real fix. The other two are optimisations of a model that currently has
nothing to optimise.

_Held-out run, router arm. Negative result, reported as run._
