# always_weak vs always_strong — qwen-7x

Prompts: **20** (dev split) · judge: `mistral.mistral-large-3-675b-instruct`, both orderings · run 20260921T015058Z-pilot-qwen-7x-always_weak-dev-calib

## Quality

| | value |
| --- | --- |
| candidate wins | 3 |
| baseline wins | 5 |
| ties | 7 |
| inconsistent (order-dependent) | 5 |
| **decided pairs** | **8 of 20** |
| candidate win rate | **37.5%** (95% CI 0.0%–75.0%) |

> The interval spans 50%, so this run does **not** distinguish the two arms. With 8 decided pairs it could not: the sample is too small to resolve anything short of a landslide.

Position-bias inconsistency: **25%** of pairs (5/20) flipped when the answers were swapped.

## Cost and latency

| metric | always_strong | always_weak |
| --- | --- | --- |
| model | `qwen.qwen3-235b-a22b-2507` | `qwen.qwen3-32b` |
| requests | 20 | 20 |
| USD per 1,000 requests | $0.3271 | $0.2255 |
| tokens per request | 411.6 | 418.6 |
| quota tokens per request | 565.0 | 569.0 |
| quota inflation vs actual | 1.37× | 1.36× |
| latency p50 | 3464 ms | 2855 ms |
| latency p90 | 5744 ms | 5346 ms |

Routing everything to the weak model would cut spend by **31%** — the ceiling on what any router can save on this pair, achieved only by giving up whatever quality the strong model adds.

_Dev-split calibration run. Not a held-out result; no router involved._
