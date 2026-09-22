#!/usr/bin/env python3
"""RouteLLM's similarity-weighted ranking router, on Bedrock features.

Their implementation (routellm/routers/routers.py, SWRankingRouter):

  1. embed the incoming prompt
  2. cosine-similarity it against every battle in the training set
  3. weight each battle by 10 * 10**(sim / max_sim)
  4. refit a weighted Bradley-Terry/Elo MLE over the whole battle set
  5. read the strong and weak tiers' Elo, convert to a win rate

Step 4 is a full logistic fit *per request* -- their class carries a
@no_parallel decorator for exactly that reason. That matters for this study:
it is not deployable inside a gateway interceptor at any sane latency, which
is a deployment finding the paper does not have to care about and we do.

**The reduction.** Our battle set is a single pair (gpt-4-1106-preview vs
mixtral-8x7b-instruct-v0.1), and for two models the Bradley-Terry MLE has a
closed form: the maximum-likelihood win probability is just the weighted
empirical win rate. So steps 3-5 collapse to

    score = sum(w_i * y_i) / sum(w_i)

which is faithful to their method on this data and vectorizes to one matmul.
Nothing is approximated; the general-case MLE is simply unnecessary here.

    .venv/bin/python analysis/sw_ranking.py --eval routellm
    .venv/bin/python analysis/sw_ranking.py --eval bbh
"""

import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cascade_sweep import auc, auc_ci  # noqa: E402
from threshold_sweep import load_embeddings  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent


def sw_scores(Q: np.ndarray, B: np.ndarray, y_b: np.ndarray,
              chunk: int = 256, tau: float = 1.0) -> tuple[np.ndarray, float]:
    """Similarity-weighted win rate for each row of Q against battle set B.

    Embeddings are L2-normalized at source (Titan `normalize: true`), so a dot
    product is already the cosine. Weighting follows their get_weightings:
    10 * 10**(sim/max_sim), where max_sim is per-query.
    """
    out = np.empty(len(Q), dtype=np.float64)
    started = time.perf_counter()
    for i in range(0, len(Q), chunk):
        sims = Q[i:i + chunk] @ B.T                       # (c, N) cosine
        mx = np.maximum(sims.max(axis=1, keepdims=True), 1e-9)
        w = 10.0 * np.power(10.0, tau * sims / mx)        # their weighting
        out[i:i + chunk] = (w @ y_b) / w.sum(axis=1)
    per_query_ms = (time.perf_counter() - started) / len(Q) * 1000
    return out, per_query_ms


def report(name, y, s, extra=""):
    lo, hi = auc_ci(y, s)
    print(f"  {name:34} AUC {auc(y, s):.3f}  [{lo:.3f}, {hi:.3f}]  {extra}")
    return {"name": name, "auc": round(auc(y, s), 3),
            "ci95": [round(lo, 3), round(hi, 3)]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", choices=["routellm", "bbh", "cascade"], default="routellm")
    ap.add_argument("--dims", type=int, default=256)
    ap.add_argument("--holdout-frac", type=float, default=0.15)
    ap.add_argument("--out", default="results/reports/sw-ranking")
    args = ap.parse_args()

    # battle set: RouteLLM's published labels, embedded with Titan (cached)
    rows = [json.loads(l) for l in
            (ROOT / "experiments/routellm_labels.jsonl").read_text().splitlines() if l.strip()]
    E = load_embeddings([r["prompt"] for r in rows], args.dims).astype(np.float32)
    yb = np.array([r["label"] for r in rows], dtype=np.float64)
    cut = int(len(rows) * (1 - args.holdout_frac))

    results, meta = [], {"eval": args.eval}
    if args.eval == "routellm":
        B, y_b = E[:cut], yb[:cut]                 # battles = training split only
        Q, y = E[cut:], yb[cut:]
        texts = [r["prompt"] for r in rows[cut:]]
        print(f"battles {len(B)}   queries {len(Q)}   positives {int(y.sum())}\n")
    else:
        path = ("experiments/bbh_labels.jsonl" if args.eval == "bbh"
                else "experiments/cascade_labels.jsonl")
        recs = [json.loads(l) for l in (ROOT / path).read_text().splitlines() if l.strip()]
        Q = load_embeddings([r["prompt"] for r in recs], args.dims).astype(np.float32)
        y = np.isin([r["tier_needed"] for r in recs], [1, 2]).astype(float)
        texts = [r["prompt"] for r in recs]
        B, y_b = E, yb                              # all 110k battles as the index
        print(f"battles {len(B)}   queries {len(Q)}   positives {int(y.sum())}\n")

    s, ms = sw_scores(Q, B, y_b)
    meta["per_query_ms_vectorised"] = round(ms, 3)
    results.append(report("SW-ranking (Titan features)", y, s, f"~{ms:.2f} ms/query"))
    results.append(report("prompt length", y,
                          np.array([float(len(t)) for t in texts])))

    out = ROOT / f"{args.out}-{args.eval}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({**meta, "results": results}, indent=2) + "\n")
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
