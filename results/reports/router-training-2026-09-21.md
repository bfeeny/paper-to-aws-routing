# Training the router on Arena preference data: a negative result that held up

The first router failed for an obvious reason — 13 training labels. This is what
happened after fixing that: 8,000 labels from Chatbot Arena human preference battles,
the same source RouteLLM used.

## The labels

`experiments/fetch_arena.py` turns battles into routing labels the way the paper does:
when a strong-tier and a weak-tier model answer the same prompt and humans pick a
winner, a strong win means "this prompt needed the strong model."

- 57,477 battles, 64 models → **12,412 cross-tier battles**, sampled to 8,000
- **47.6% strong-needed** — nearly balanced, unlike our MT-Bench-derived labels (63/37)
- Source pinned by SHA-256; tier map is explicit in the script, because it is the most
  consequential judgment in the pipeline

## Every lever, and what it bought

Features are Titan v2 embeddings of the prompt; the target is the routing label.

| Setup | Train AUC | **Hold-out AUC** |
| --- | --- | --- |
| 256-d, 8,000 battles, mixed pairings | 0.618 | **0.525** |
| 256-d, 2,465 battles, GPT-4 vs GPT-3.5 only | 0.704 | **0.520** |
| 1024-d, 8,000 battles | 0.705 | **0.495** |
| MLP (64 hidden units), 256-d | 0.606 | **0.524** |
| L2 from 0.01 to 10 | 0.615–0.618 | **0.525** (flat) |
| **Prompt length alone, no model at all** | — | **0.530** |

Chance is 0.500.

Read the last row against the others. **A single scalar — how long the prompt is —
matches or beats every learned model we fitted.** More dimensions, more capacity, a
homogeneous model pairing, three orders of magnitude of regularization: each one
improves the *training* fit and none moves hold-out performance off chance.

That pattern is diagnostic. When capacity increases fit but never generalization, the
features do not contain the signal. This is not a tuning problem.

## What this does and doesn't say

**It does not say routing cannot work.** RouteLLM reports real gains, and two
differences plausibly explain the gap:

1. **Scale.** They trained on roughly 80,000 battles; the public 55k dataset yields
   12,412 cross-tier battles, and any single model pairing has only 350–700.
2. **Augmentation.** The paper is explicit that preference data alone was insufficient
   and that augmented labels were needed to reach the reported numbers. We used the raw
   data only.

**It does say the hard part isn't the plumbing.** The gateway, interceptor, ledger,
judge and deployment all work; a router can be trained, shipped as JSON weights and
scored inside a Lambda in ~110 ms. The part that doesn't come for free is a prompt-only
signal that predicts which model a request needs — and reproducing it takes more than
downloading a preference dataset and fitting a classifier to it.

**A practical corollary for anyone building this:** before deploying a learned router,
check it against prompt length. If a scalar you can compute for free does as well, the
model is decoration with a latency cost attached.

## Where this goes next

1. **Augment the labels**, as the paper did — judge additional prompts with a strong
   model to manufacture training signal beyond the Arena battles.
2. **Sharpen the target.** "Strong won a human vote" is noisy: human preference on
   open-ended chat is not the same question as "would the weak answer have been good
   enough here," which is what a router actually needs.
3. **Consider that the honest answer may be a heuristic.** If length-based routing
   captures most of the achievable saving on a pair with a 92% price gap, that is a
   useful result to publish, and cheaper to operate than an embedding call per request.

_All numbers reproducible from committed labels and cached embeddings; see
`analysis/train_router.py`._
