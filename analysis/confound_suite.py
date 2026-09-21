#!/usr/bin/env python3
"""The three checks that decide whether an in-distribution AUC means anything.

Each one exists because a headline number in this study did not survive it.

  within-group label shuffle  permute labels inside each group, preserving
                              group base rates and destroying everything else.
                              Whatever AUC survives was group lookup.
  leave-one-group-out         the unit of analysis is the group, not the item;
                              a single random held-out split is one draw from a
                              distribution wide enough to invert the conclusion.
  paired cluster bootstrap    resample groups, not items, and compare router to
                              the group-base-rate oracle on the same resample.

    .venv/bin/python analysis/confound_suite.py --labels experiments/bbh_labels.jsonl \
        --items experiments/bbh_items.jsonl --group subject
"""
import argparse, json, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cascade_sweep import auc
from train_router import embed, fit_logistic, scores

ROOT = pathlib.Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="experiments/bbh_labels.jsonl")
    ap.add_argument("--items", default="experiments/bbh_items.jsonl")
    ap.add_argument("--group", default="subject")
    ap.add_argument("--min-pos", type=int, default=5)
    ap.add_argument("--out", default="results/reports/confounds")
    args = ap.parse_args()

    recs = [json.loads(l) for l in (ROOT / args.labels).read_text().splitlines() if l.strip()]
    items = {json.loads(l)["id"]: json.loads(l)
             for l in (ROOT / args.items).read_text().splitlines() if l.strip()}
    recs = [r for r in recs if r["id"] in items]
    g = np.array([str(items[r["id"]].get(args.group)) for r in recs])
    y = np.isin([r["tier_needed"] for r in recs], [1, 2]).astype(float)
    X = embed([r["prompt"] for r in recs], "personal", "us-east-1", 256, workers=24)
    L = np.array([float(len(r["prompt"])) for r in recs])
    print(f"n={len(recs)} groups={len(set(g))} positives={int(y.sum())}")

    rng = np.random.default_rng(20260921)
    perm = rng.permutation(len(recs)); cut = int(len(recs) * 0.70)
    tr, te = perm[:cut], perm[cut:]
    w, b = fit_logistic(X[tr], y[tr], l2=1.0)
    real = auc(y[te], scores(X[te], w, b))

    # 1. within-group shuffle
    ys = y.copy()
    for grp in set(g):
        m = np.where(g == grp)[0]
        ys[m] = rng.permutation(y[m])
    w2, b2 = fit_logistic(X[tr], ys[tr], l2=1.0)
    shuf = auc(y[te], scores(X[te], w2, b2))
    print(f"\nwithin-group label shuffle: {shuf:.3f}  vs real labels {real:.3f}")
    print(f"  -> {shuf / real:.0%} of the in-distribution AUC is group lookup")

    # 2. leave-one-group-out
    loo = []
    for grp in sorted(set(g)):
        m = g == grp
        if y[m].sum() < args.min_pos or (1 - y[m]).sum() < args.min_pos:
            continue
        w3, b3 = fit_logistic(X[~m], y[~m], l2=1.0)
        loo.append((grp, auc(y[m], scores(X[m], w3, b3)), auc(y[m], L[m])))
    r_m = float(np.mean([a for _, a, _ in loo])); l_m = float(np.mean([l for _, _, l in loo]))
    print(f"\nleave-one-group-out over {len(loo)} groups "
          f"(skipped {len(set(g)) - len(loo)} with <{args.min_pos} of a class):")
    print(f"  router mean {r_m:.3f} (sd {np.std([a for _,a,_ in loo]):.3f}, "
          f"range {min(a for _,a,_ in loo):.2f}-{max(a for _,a,_ in loo):.2f})")
    print(f"  length mean {l_m:.3f}")

    # 3. paired cluster bootstrap, router minus group-rate oracle
    rate = {t: y[tr][g[tr] == t].mean() for t in set(g)}
    orc = np.array([rate[t] for t in g])
    s_te, o_te, g_te, y_te = scores(X[te], w, b), orc[te], g[te], y[te]
    groups = sorted(set(g_te)); diffs = []
    for _ in range(2000):
        pick = rng.choice(groups, len(groups), replace=True)
        idx = np.concatenate([np.where(g_te == p)[0] for p in pick])
        if y_te[idx].sum() in (0, len(idx)):
            continue
        diffs.append(auc(y_te[idx], s_te[idx]) - auc(y_te[idx], o_te[idx]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"\npaired cluster bootstrap (router - group oracle): "
          f"{np.mean(diffs):+.3f}  [{lo:+.3f}, {hi:+.3f}]")
    print("  -> " + ("indistinguishable from the group base rate"
                     if lo < 0 < hi else "a real difference"))

    out = ROOT / f"{args.out}-{pathlib.Path(args.labels).stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "labels": args.labels, "group_field": args.group, "n": len(recs),
        "auc_real": round(real, 3), "auc_within_group_shuffle": round(shuf, 3),
        "loo": [{"group": a, "router": round(b_, 3), "length": round(c, 3)} for a, b_, c in loo],
        "loo_mean_router": round(r_m, 3), "loo_mean_length": round(l_m, 3),
        "paired_cluster_bootstrap_router_minus_oracle":
            {"mean": round(float(np.mean(diffs)), 3), "ci95": [round(lo, 3), round(hi, 3)]},
    }, indent=2) + "\n")
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
