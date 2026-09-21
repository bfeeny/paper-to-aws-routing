#!/usr/bin/env python3
"""What the resampled cascade says about the reliability of a single-draw label.

Four questions, in increasing order of how much they matter:

  pass rates      how often each tier actually answers an item correctly
  test-retest     two independent draws at temperature 0, same prompt, same
                  model: do they agree? Cohen's kappa on the binary outcome.
  tier churn      rebuild `tier_needed` from each replicate separately. Every
                  change is an item whose published label was a coin flip.
  ceiling         the soft label P(correct) is the best difficulty signal that
                  exists for these items. Scoring it against a single-draw
                  label gives the maximum AUC any predictor can reach -- so
                  every AUC in the study should be read against this, not 1.0.

    .venv/bin/python analysis/stability_report.py
"""
import collections, json, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cascade_sweep import auc, auc_ci

ROOT = pathlib.Path(__file__).resolve().parent.parent
TIERS = ["haiku", "sonnet", "opus"]


def kappa(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return (po - pe) / (1 - pe) if pe < 1 else float("nan"), po


def main() -> int:
    raw = [json.loads(l) for l in
           (ROOT / "experiments/stability_raw.jsonl").read_text().splitlines() if l.strip()]
    draws = collections.defaultdict(list)          # (id, tier, temp) -> [bool,...]
    for r in raw:
        draws[(r["id"], r["tier"], r["temp"])].append(bool(r["correct"]))
    ids = sorted({k[0] for k in draws})
    print(f"items {len(ids)}   raw calls {len(raw)}")

    print("\npass rate by tier (mean over draws):")
    for t in TIERS:
        for temp in (0.0, 1.0):
            vals = [np.mean(v) for k, v in draws.items() if k[1] == t and k[2] == temp and v]
            if vals:
                print(f"  {t:7} T={temp:.0f}  {np.mean(vals):.3f}")

    print("\ntest-retest at T=0 (draw 1 vs draw 2, identical configuration):")
    for t in TIERS:
        a, b = [], []
        for i in ids:
            v = draws.get((i, t, 0.0), [])
            if len(v) >= 2:
                a.append(v[0]); b.append(v[1])
        if len(a) >= 20:
            k, po = kappa(a, b)
            print(f"  {t:7} n={len(a):4}  agreement {po:.1%}   kappa {k:.3f}")

    # how unstable is the derived tier label?
    def tier_from(i, idx, temp):
        for j, t in enumerate(TIERS):
            v = draws.get((i, t, temp), [])
            if idx < len(v) and v[idx]:
                return j
        return len(TIERS)

    for temp in (0.0, 1.0):
        n_draws = min(len(v) for k, v in draws.items() if k[2] == temp) if draws else 0
        rows = [[tier_from(i, d, temp) for d in range(n_draws)] for i in ids]
        unstable = sum(len(set(r)) > 1 for r in rows)
        esc = [[1 if t in (1, 2) else 0 for t in r] for r in rows]
        esc_unstable = sum(len(set(e)) > 1 for e in esc)
        print(f"\ntier label rebuilt from each of {n_draws} replicates, T={temp:.0f}:")
        print(f"  tier_needed changes on        {unstable}/{len(ids)} items ({unstable/len(ids):.1%})")
        print(f"  binary escalation label flips {esc_unstable}/{len(ids)} items "
              f"({esc_unstable/len(ids):.1%})")

    # attenuation ceiling
    soft, obs = [], []
    for i in ids:
        p_esc = np.mean([1 if tier_from(i, d, 1.0) in (1, 2) else 0
                         for d in range(min(4, len(draws.get((i, "haiku", 1.0), []))))]) \
            if draws.get((i, "haiku", 1.0)) else None
        single = 1 if tier_from(i, 0, 0.0) in (1, 2) else 0
        if p_esc is not None:
            soft.append(p_esc); obs.append(single)
    soft, obs = np.array(soft), np.array(obs, float)
    if 0 < obs.sum() < len(obs):
        a = auc(obs, soft); lo, hi = auc_ci(obs, soft)
        print(f"\nCEILING — soft P(escalate) scored against a single-draw label:")
        print(f"  AUC {a:.3f}  [{lo:.3f}, {hi:.3f}]   n={len(obs)}, positives={int(obs.sum())}")
        print("  No predictor of any kind can exceed this against these labels.")
        out = {"ceiling_auc": round(a, 3), "ceiling_ci95": [round(lo, 3), round(hi, 3)],
               "n": int(len(obs)), "positives": int(obs.sum())}
        p = ROOT / "results/reports/label-stability.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2) + "\n")
        print(f"\nwrote {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
