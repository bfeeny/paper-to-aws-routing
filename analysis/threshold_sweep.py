#!/usr/bin/env python3
"""Sweep the router's threshold and report cost against quality.

A single AUC says the ranking is informative; it does not say what to deploy.
The operating question is: if I send the top c% of prompts to the strong model,
what fraction of the quality gap do I recover, and what does the bill look like?

Metrics follow RouteLLM so the numbers are directly comparable to the paper:

    PGR(c)   performance gap recovered at call rate c
             (P(c) - P_weak) / (P_strong - P_weak)
    APGR     average PGR across all call rates -- one number for a whole router
    CPT(x)   call-performance threshold: the smallest c reaching PGR >= x

For a binary "did the weak model suffice" label, PGR(c) reduces exactly to
recall of the strong-needed class, which is worth knowing: it means the chart
is a recall curve wearing a cost axis.

Two baselines are plotted, and the second is the one that matters. Random
routing is the floor nobody claims to beat. **Prompt length** is the bar that
actually has to be cleared, because it is free -- if a learned router does not
beat `len(prompt)`, it is latency and complexity with no payoff.

    python3 analysis/threshold_sweep.py --labels experiments/routellm_labels.jsonl
"""

import argparse
import hashlib
import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "experiments" / "cache"
EMBED_MODEL = "amazon.titan-embed-text-v2:0"
DIMS = 256


def load_embeddings(texts: list[str], dims: int) -> np.ndarray:
    """Read the embeddings training already paid for.

    Deliberately offline: this is pure analysis, so it must not need botocore,
    credentials or a network. If the cache is missing, that is a real error --
    silently re-embedding would charge for a chart.
    """
    key = hashlib.sha256(("|".join(texts)).encode()).hexdigest()[:16]
    cache = CACHE_DIR / f"emb-{EMBED_MODEL.replace(':', '_')}-{dims}-{key}.npy"
    if not cache.exists():
        raise SystemExit(
            f"no cached embeddings at {cache.relative_to(ROOT)}.\n"
            f"Run analysis/train_router.py on the same --labels and --dims first."
        )
    return np.load(cache)


def scores(X, w, b):
    return 1 / (1 + np.exp(-(X @ w + b)))

# Bedrock on-demand list prices, USD per 1M tokens, us-east-1.
# These are the one input here most likely to drift -- re-check before publishing.
PRICES = {
    "claude-haiku-4-5": {"in": 1.00, "out": 5.00},
    "claude-sonnet-4-6": {"in": 3.00, "out": 15.00},
    "claude-opus-4-5": {"in": 5.00, "out": 25.00},
}


def pgr_curve(y: np.ndarray, score: np.ndarray, points: int = 201):
    """PGR at every call rate, by routing the highest-scoring prompts first.

    Ties are broken by the sort, which is the same rule a deployed threshold
    would apply, so the curve is achievable rather than optimistic.
    """
    order = np.argsort(-score)           # most likely to need strong, first
    y_sorted = y[order]
    cum_tp = np.concatenate([[0.0], np.cumsum(y_sorted)])
    n, n1 = len(y), float(y.sum())
    rates = np.linspace(0.0, 1.0, points)
    ks = np.round(rates * n).astype(int)
    pgr = cum_tp[ks] / n1 if n1 else np.zeros_like(rates)
    return rates, pgr


def cpt(rates: np.ndarray, pgr: np.ndarray, target: float) -> float | None:
    """Smallest call rate reaching the target PGR, or None if never reached."""
    hit = np.nonzero(pgr >= target)[0]
    return float(rates[hit[0]]) if len(hit) else None


def cost_per_1k(call_rate: float, strong: str, weak: str,
                in_tok: int, out_tok: int) -> float:
    def unit(model):
        p = PRICES[model]
        return (in_tok * p["in"] + out_tok * p["out"]) / 1e6
    return 1000 * (call_rate * unit(strong) + (1 - call_rate) * unit(weak))


def summarize(name, y, score, strong, weak, in_tok, out_tok) -> dict:
    rates, pgr = pgr_curve(y, score)
    apgr = float(np.trapezoid(pgr, rates))
    row = {
        "router": name,
        "apgr": round(apgr, 4),
        "cpt_50": cpt(rates, pgr, 0.50),
        "cpt_80": cpt(rates, pgr, 0.80),
        "cpt_90": cpt(rates, pgr, 0.90),
        "operating_points": [],
    }
    always_strong = cost_per_1k(1.0, strong, weak, in_tok, out_tok)
    for target in (0.50, 0.80, 0.90):
        c = cpt(rates, pgr, target)
        if c is None:
            continue
        cost = cost_per_1k(c, strong, weak, in_tok, out_tok)
        row["operating_points"].append({
            "pgr_target": target,
            "call_rate": round(c, 4),
            "cost_per_1k_requests": round(cost, 3),
            "saving_vs_always_strong": round(1 - cost / always_strong, 4),
        })
    return row, rates, pgr


def chart(curves: dict, out_path: pathlib.Path, strong: str, weak: str,
          in_tok: int, out_tok: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
        "figure.dpi": 160,
    })
    style = {
        "router": {"color": "#1f6fb2", "lw": 2.0, "zorder": 3},
        "prompt length": {"color": "#a8434f", "lw": 1.6, "ls": "--", "zorder": 2},
        "random": {"color": "#8a8a8a", "lw": 1.2, "ls": ":", "zorder": 1},
    }
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.8))

    for name, (rates, pgr) in curves.items():
        ax1.plot(rates * 100, pgr * 100, label=name, **style.get(name, {}))
    ax1.axhline(80, color="#333", lw=0.8, ls="-.", alpha=0.6)
    ax1.text(2, 81.5, "80% of the gap", fontsize=7.5, color="#333")
    ax1.set_xlabel(f"calls routed to {strong} (%)")
    ax1.set_ylabel("performance gap recovered (%)")
    ax1.set_title("Quality recovered per call to the strong model", fontsize=10, pad=8)
    ax1.set_xlim(0, 100); ax1.set_ylim(0, 100)
    ax1.legend(frameon=False, fontsize=8, loc="lower right")

    for name, (rates, pgr) in curves.items():
        costs = [cost_per_1k(c, strong, weak, in_tok, out_tok) for c in rates]
        ax2.plot(costs, pgr * 100, label=name, **style.get(name, {}))
    full = cost_per_1k(1.0, strong, weak, in_tok, out_tok)
    ax2.axvline(full, color="#333", lw=0.8, ls="-.", alpha=0.6)
    ax2.text(full, 6, f"  always {strong}\n  ${full:,.2f}", fontsize=7.5, color="#333",
             ha="right", va="bottom")
    ax2.set_xlabel("cost per 1,000 requests (USD)")
    ax2.set_ylabel("performance gap recovered (%)")
    ax2.set_title("What the quality actually costs", fontsize=10, pad=8)
    ax2.set_ylim(0, 100)

    fig.suptitle(f"Routing between {strong} and {weak}", fontsize=11, y=1.0)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"wrote {out_path.relative_to(ROOT)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="experiments/routellm_labels.jsonl")
    ap.add_argument("--weights", default="router/artifacts/router_weights.json")
    ap.add_argument("--holdout-frac", type=float, default=0.15)
    ap.add_argument("--dims", type=int, default=DIMS)
    ap.add_argument("--strong", default="claude-opus-4-5", choices=list(PRICES))
    ap.add_argument("--weak", default="claude-haiku-4-5", choices=list(PRICES))
    ap.add_argument("--in-tokens", type=int, default=140, help="avg input tokens/request")
    ap.add_argument("--out-tokens", type=int, default=280, help="avg output tokens/request")
    ap.add_argument("--out", default="results/reports/threshold-sweep")
    args = ap.parse_args()

    rows = [json.loads(l) for l in (ROOT / args.labels).read_text().splitlines() if l.strip()]
    art = json.loads((ROOT / args.weights).read_text())
    w, b = np.array(art["weights"], dtype=float), float(art["bias"])

    # Same split as training: the sweep must never see a trained-on prompt.
    X = load_embeddings([r["prompt"] for r in rows], args.dims)
    y = np.array([r["label"] for r in rows], dtype=float)
    cut = int(len(rows) * (1 - args.holdout_frac))
    Xte, yte = X[cut:], y[cut:]
    prompts_te = [r["prompt"] for r in rows[cut:]]

    rng = np.random.default_rng(20260921)
    candidates = {
        "router": scores(Xte, w, b),
        "prompt length": np.array([float(len(p)) for p in prompts_te]),
        "random": rng.random(len(yte)),
    }

    report, curves = [], {}
    for name, s in candidates.items():
        row, rates, pgr = summarize(name, yte, s, args.strong, args.weak,
                                    args.in_tokens, args.out_tokens)
        report.append(row)
        curves[name] = (rates, pgr)

    out_json = ROOT / f"{args.out}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "labels": args.labels,
        "holdout_n": int(len(yte)),
        "strong_needed_rate": round(float(yte.mean()), 4),
        "priced_as": {"strong": args.strong, "weak": args.weak,
                      "avg_in_tokens": args.in_tokens, "avg_out_tokens": args.out_tokens,
                      "prices_usd_per_1m": PRICES},
        "results": report,
    }
    out_json.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"hold-out n={len(yte)}  strong-needed={yte.mean():.1%}\n")
    print(f"{'router':16} {'APGR':>7} {'CPT(50%)':>9} {'CPT(80%)':>9} {'CPT(90%)':>9}")
    for r in report:
        f = lambda v: f"{v:.1%}" if v is not None else "never"  # noqa: E731
        print(f"{r['router']:16} {r['apgr']:7.3f} {f(r['cpt_50']):>9} "
              f"{f(r['cpt_80']):>9} {f(r['cpt_90']):>9}")

    print(f"\ncost per 1,000 requests, routing {args.weak} -> {args.strong}")
    for r in report:
        for op in r["operating_points"]:
            print(f"  {r['router']:16} PGR {op['pgr_target']:.0%}: "
                  f"{op['call_rate']:6.1%} of calls  "
                  f"${op['cost_per_1k_requests']:8.2f}  "
                  f"({op['saving_vs_always_strong']:.1%} under always-strong)")

    chart(curves, ROOT / f"{args.out}.png", args.strong, args.weak,
          args.in_tokens, args.out_tokens)
    print(f"wrote {out_json.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
