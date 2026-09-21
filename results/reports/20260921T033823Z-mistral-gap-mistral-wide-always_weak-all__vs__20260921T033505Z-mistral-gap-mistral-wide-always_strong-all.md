# always_weak vs always_strong — mistral-wide

Prompts: **80** (all split) · judge: `qwen.qwen3-235b-a22b-2507`, both orderings · run 20260921T033823Z-mistral-gap-mistral-wide-always_weak-all

## Quality

| | value |
| --- | --- |
| candidate wins | 5 |
| baseline wins | 34 |
| ties | 12 |
| inconsistent (order-dependent) | 29 |
| **decided pairs** | **39 of 80** |
| candidate win rate | **12.8%** (95% CI 2.9%–24.3%) |

Position-bias inconsistency: **36%** of pairs (29/80) flipped when the answers were swapped.

## Cost and latency

| metric | always_strong | always_weak |
| --- | --- | --- |
| model | `mistral.mistral-large-3-675b-instruct` | `mistral.ministral-3-3b-instruct` |
| requests | 80 | 80 |
| USD per 1,000 requests | $0.5766 | $0.0456 |
| tokens per request | 432.4 | 456.6 |
| quota tokens per request | 584.0 | 584.0 |
| quota inflation vs actual | 1.35× | 1.28× |
| latency p50 | 2386 ms | 1539 ms |
| latency p90 | 3822 ms | 1630 ms |

Routing everything to the weak model would cut spend by **92%** — the ceiling on what any router can save on this pair, achieved only by giving up whatever quality the strong model adds.

_Full-set calibration run (no router involved)._
