#!/usr/bin/env python3
"""Threshold sweep against the Claude cascade labels.

Two questions, and the first is the one the study exists to answer.

**Transfer.** The router trained on RouteLLM's 110k labels learned one model
pair: GPT-4 against Mixtral, circa 2023. Does that transfer to a different pair
-- Claude Haiku against Sonnet/Opus, on our own graded benchmarks? An earlier
attempt said chance, but at n=51. This asks the same question at n=2,709.

**In-distribution.** Train a router on the cascade labels themselves. This is
the ceiling the transfer result should be read against: if a router fitted to
these very prompts also fails, the problem is the features, not the mismatch.

The target is decision-theoretic rather than descriptive: positive means
"escalating above the cheap tier changes the outcome". Items no tier answers
are labeled 0, because paying for Opus when Opus also fails is not a win --
the alternative coding is reported alongside so the choice is visible.

    .venv/bin/python analysis/cascade_sweep.py
"""

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from threshold_sweep import PRICES, cost_per_1k, cpt, pgr_curve  # noqa: E402
from train_router import embed, fit_logistic, scores  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent


def auc(y: np.ndarray, s: np.ndarray) -> float:
    order = np.argsort(s)
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    n1, n0 = float(y.sum()), float((1 - y).sum())
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def auc_ci(y, s, boots=2000, seed=20260921):
    """Percentile bootstrap. At this n the interval is the result, not decoration."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(boots):
        idx = rng.integers(0, len(y), len(y))
        yb = y[idx]
        if yb.sum() in (0, len(yb)):
            continue
        vals.append(auc(yb, s[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="experiments/cascade_labels.jsonl")
    ap.add_argument("--weights", default="router/artifacts/router_weights.json")
    ap.add_argument("--dims", type=int, default=256)
    ap.add_argument("--holdout-frac", type=float, default=0.30)
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--strong", default="claude-opus-4-5", choices=list(PRICES))
    ap.add_argument("--weak", default="claude-haiku-4-5", choices=list(PRICES))
    ap.add_argument("--in-tokens", type=int, default=140)
    ap.add_argument("--out-tokens", type=int, default=280)
    ap.add_argument("--out", default="results/reports/cascade-sweep")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--profile", default="personal")
    ap.add_argument("--region", default="us-east-1")
    args = ap.parse_args()

    recs = [json.loads(l) for l in (ROOT / args.labels).read_text().splitlines() if l.strip()]
    prompts = [r["prompt"] for r in recs]
    tier = np.array([r["tier_needed"] for r in recs])

    # primary coding: escalation changes the outcome
    y = np.isin(tier, [1, 2]).astype(float)
    # alternative: "haiku was not enough", counting unanswerable as positive
    y_alt = (tier >= 1).astype(float)

    print(f"items: {len(recs)}   escalation-worthy: {y.mean():.1%}   "
          f"(alt coding, 'haiku insufficient': {y_alt.mean():.1%})")

    X = embed(prompts, args.profile, args.region, args.dims, workers=args.workers)
    length = np.array([float(len(p)) for p in prompts])

    art = json.loads((ROOT / args.weights).read_text())
    w_tr, b_tr = np.array(art["weights"], dtype=float), float(art["bias"])
    transfer = scores(X, w_tr, b_tr)

    # in-distribution router, fitted on a disjoint slice of the same set
    rng = np.random.default_rng(20260921)
    perm = rng.permutation(len(recs))
    cut = int(len(recs) * (1 - args.holdout_frac))
    tr, te = perm[:cut], perm[cut:]
    w_in, b_in = fit_logistic(X[tr], y[tr], l2=args.l2)

    results = []

    def record(name, yy, ss, note=""):
        lo, hi = auc_ci(yy, ss)
        a = auc(yy, ss)
        rates, pgr = pgr_curve(yy, ss)
        row = {"router": name, "n": int(len(yy)), "positives": int(yy.sum()),
               "auc": round(a, 3), "auc_ci95": [round(lo, 3), round(hi, 3)],
               "apgr": round(float(np.trapezoid(pgr, rates)), 4),
               "cpt_50": cpt(rates, pgr, 0.50), "cpt_80": cpt(rates, pgr, 0.80),
               "note": note}
        results.append(row)
        return rates, pgr

    curves = {}
    curves["transfer (110k GPT-4/Mixtral)"] = record(
        "transfer (110k GPT-4/Mixtral)", y, transfer, "full cascade set, never trained on it")
    curves["prompt length"] = record("prompt length", y, length, "free baseline, full set")
    curves["in-distribution (cascade)"] = record(
        "in-distribution (cascade)", y[te], scores(X[te], w_in, b_in),
        f"trained on {len(tr)}, scored on {len(te)} held out")
    record("prompt length (same hold-out)", y[te], length[te], "comparable to the row above")
    record("transfer, alt coding", y_alt, transfer, "unanswerable counted as positive")

    print(f"\n{'router':34}{'n':>6}{'pos':>6}{'AUC':>7}  {'95% CI':>16}{'APGR':>8}")
    for r in results:
        ci = f"[{r['auc_ci95'][0]:.3f}, {r['auc_ci95'][1]:.3f}]"
        print(f"{r['router']:34}{r['n']:6}{r['positives']:6}{r['auc']:7.3f}  {ci:>16}{r['apgr']:8.3f}")

    print(f"\ncost per 1,000 requests, {args.weak} -> {args.strong}")
    always = cost_per_1k(1.0, args.strong, args.weak, args.in_tokens, args.out_tokens)
    for r in results:
        for target, key in ((0.50, "cpt_50"), (0.80, "cpt_80")):
            c = r[key]
            if c is None:
                continue
            cost = cost_per_1k(c, args.strong, args.weak, args.in_tokens, args.out_tokens)
            print(f"  {r['router']:34} PGR {target:.0%}: {c:6.1%} of calls  "
                  f"${cost:6.2f}  ({1 - cost / always:.1%} under always-strong)")

    # --- confound check -------------------------------------------------
    #
    # The benchmarks differ enormously in how often escalation helps (GSM8K
    # 1.9%, BBH 25.2%). A model that only learns to recognize *which benchmark
    # a prompt came from* would post a strong AUC while knowing nothing about
    # difficulty. Deployments see one traffic distribution, not five, so that
    # skill would not survive contact with production.
    bm = np.array([r["benchmark"] for r in recs])
    rate = {b: float(y[tr][bm[tr] == b].mean()) for b in sorted(set(bm))}
    oracle = np.array([rate[b] for b in bm])
    o_auc, (o_lo, o_hi) = auc(y[te], oracle[te]), auc_ci(y[te], oracle[te])
    s_in = scores(X, w_in, b_in)

    within = []
    for bname in sorted(set(bm)):
        m = bm[te] == bname
        yy, ss = y[te][m], s_in[te][m]
        if yy.sum() < 8 or (1 - yy).sum() < 8:
            within.append({"benchmark": bname, "n": int(m.sum()),
                           "positives": int(yy.sum()), "auc": None,
                           "note": "too few positives to estimate"})
            continue
        lo, hi = auc_ci(yy, ss)
        within.append({"benchmark": bname, "n": int(m.sum()),
                       "positives": int(yy.sum()), "auc": round(auc(yy, ss), 3),
                       "auc_ci95": [round(lo, 3), round(hi, 3)]})

    print(f"\nconfound: benchmark-identity oracle  AUC {o_auc:.3f} "
          f"[{o_lo:.3f}, {o_hi:.3f}]  (router scored {results[2]['auc']:.3f})")
    print("escalation rate by benchmark:",
          "  ".join(f"{b} {v:.1%}" for b, v in sorted(rate.items(), key=lambda kv: -kv[1])))
    print("\nwithin-benchmark AUC (style held constant — the honest test):")
    for r in within:
        if r["auc"] is None:
            print(f"  {r['benchmark']:10} n={r['n']:4} pos={r['positives']:3}  {r['note']}")
        else:
            print(f"  {r['benchmark']:10} n={r['n']:4} pos={r['positives']:3}  "
                  f"AUC {r['auc']:.3f}  [{r['auc_ci95'][0]:.3f}, {r['auc_ci95'][1]:.3f}]")

    out_json = ROOT / f"{args.out}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({
        "labels": args.labels, "n": len(recs),
        "escalation_rate": round(float(y.mean()), 4),
        "priced_as": {"strong": args.strong, "weak": args.weak,
                      "avg_in_tokens": args.in_tokens, "avg_out_tokens": args.out_tokens},
        "results": results,
        "confound_benchmark_identity": {
            "oracle_auc": round(o_auc, 3), "oracle_ci95": [round(o_lo, 3), round(o_hi, 3)],
            "escalation_rate_by_benchmark": {k: round(v, 4) for k, v in rate.items()},
            "within_benchmark": within,
        },
    }, indent=2) + "\n")

    _chart(curves, ROOT / f"{args.out}.png", args.strong, args.weak,
           args.in_tokens, args.out_tokens)
    print(f"wrote {out_json.relative_to(ROOT)}")
    return 0


def _chart(curves, out_path, strong, weak, in_tok, out_tok):
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
        "in-distribution (cascade)": {"color": "#1f6fb2", "lw": 2.0, "zorder": 3},
        "prompt length": {"color": "#a8434f", "lw": 1.6, "ls": "--", "zorder": 2},
        "transfer (110k GPT-4/Mixtral)": {"color": "#9a6b1f", "lw": 1.8, "ls": "-", "zorder": 2},
    }
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for name, (rates, pgr) in curves.items():
        ax.plot(rates * 100, pgr * 100, label=name, **style.get(name, {}))
    ax.plot([0, 100], [0, 100], color="#8a8a8a", lw=1.0, ls=":", label="random", zorder=1)
    ax.set_xlabel(f"calls escalated above {weak} (%)")
    ax.set_ylabel("escalation-worthy prompts caught (%)")
    ax.set_title("Does a router trained on one model pair\ntransfer to another?",
                 fontsize=10, pad=8)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"wrote {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    raise SystemExit(main())
