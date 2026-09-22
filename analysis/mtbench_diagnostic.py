#!/usr/bin/env python3
"""Run our confound diagnostic on RouteLLM's own MT-Bench evaluation set.

MT-Bench is where the paper's headline gains live, and it is stratified by
construction: 8 categories, 10 questions each, with very different strong-win
rates. A router that recognizes "this is a coding question" and recalls that
category's base rate will post gains without modeling difficulty at all.

The paper evaluates on item-level splits. It does not report a category
ablation, so the question has not been asked. It is cheap to ask: both the
questions (with categories) and the GPT-4 judgments are published in the repo.

The label is theirs: for each (question, turn), strong-needed = the judge gave
GPT-4 a higher score than Mixtral.

This does not re-run their router -- the released checkpoints consume OpenAI
text-embedding-3-small features. It asks the prior question: how much signal
is available from category identity alone? Whatever that is, it is an upper
bound on what "recognizes the traffic mix" could be worth here.

    .venv/bin/python analysis/mtbench_diagnostic.py
"""
import collections, json, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cascade_sweep import auc, auc_ci
from train_router import embed, fit_logistic, scores
from threshold_sweep import load_embeddings

ROOT = pathlib.Path(__file__).resolve().parent.parent
CACHE = ROOT / "experiments" / "cache"
STRONG, WEAK = "gpt-4-1106-preview", "mistralai/Mixtral-8x7B-Instruct-v0.1"


def main() -> int:
    qs = {str(json.loads(l)["question_id"]): json.loads(l)
          for l in (CACHE / "mtbench_question.jsonl").read_text().splitlines() if l.strip()}
    sc = collections.defaultdict(dict)
    for l in (CACHE / "mtbench_judgements.jsonl").read_text().splitlines():
        if not l.strip():
            continue
        j = json.loads(l)
        try:
            sc[(str(j["question_id"]), int(j["turn"]))][j["model"]] = float(j["score"])
        except (ValueError, KeyError):
            continue

    rows = []
    for (qid, turn), m in sorted(sc.items()):
        if STRONG not in m or WEAK not in m or qid not in qs:
            continue
        q = qs[qid]
        turns = q["turns"]
        rows.append({"qid": qid, "turn": turn, "category": q["category"],
                     "prompt": turns[min(turn, len(turns)) - 1],
                     "label": 1.0 if m[STRONG] > m[WEAK] else 0.0,
                     "margin": m[STRONG] - m[WEAK]})
    y = np.array([r["label"] for r in rows])
    cat = np.array([r["category"] for r in rows])
    print(f"MT-Bench eval set: n={len(rows)} items "
          f"({len(set(r['qid'] for r in rows))} questions x 2 turns), "
          f"strong-needed {y.mean():.1%}")

    print("\nstrong-needed rate by category:")
    rate = {}
    for c in sorted(set(cat)):
        rate[c] = float(y[cat == c].mean())
        print(f"  {c:14} {rate[c]:6.1%}   n={int((cat == c).sum())}")
    spread = max(rate.values()) - min(rate.values())
    print(f"  spread: {spread:.1%}")

    orc = np.array([rate[c] for c in cat])
    lo, hi = auc_ci(y, orc)
    print(f"\nCATEGORY-IDENTITY ORACLE: AUC {auc(y, orc):.3f}  [{lo:.3f}, {hi:.3f}]")
    print("  A router that only recognizes the category reaches this, knowing")
    print("  nothing about difficulty. It is the bar their gains must clear.")

    X = embed([r["prompt"] for r in rows], "personal", "us-east-1", 256, workers=16)
    L = np.array([float(len(r["prompt"])) for r in rows])

    # our 110k-trained router, transferred
    art = json.loads((ROOT / "router/artifacts/router_weights.json").read_text())
    w, b = np.array(art["weights"]), float(art["bias"])
    s_tr = scores(X, w, b)
    lo2, hi2 = auc_ci(y, s_tr)
    print(f"\nour 110k router (same pair, transferred): AUC {auc(y, s_tr):.3f} "
          f"[{lo2:.3f}, {hi2:.3f}]")
    lo3, hi3 = auc_ci(y, L)
    print(f"prompt length:                           AUC {auc(y, L):.3f} "
          f"[{lo3:.3f}, {hi3:.3f}]")

    # leave-one-category-out on a router fitted to MT-Bench itself
    loco = []
    for c in sorted(set(cat)):
        m = cat == c
        if y[m].sum() < 3 or (1 - y[m]).sum() < 3:
            continue
        w2, b2 = fit_logistic(X[~m], y[~m], l2=1.0)
        loco.append((c, auc(y[m], scores(X[m], w2, b2))))
    if loco:
        print(f"\nleave-one-category-out (fitted on MT-Bench, {len(loco)} folds):")
        for c, a in loco:
            print(f"  {c:14} {a:.3f}")
        print(f"  {'MEAN':14} {np.mean([a for _, a in loco]):.3f}")

    out = ROOT / "results/reports/mtbench-diagnostic.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "n": len(rows), "strong_needed_rate": round(float(y.mean()), 4),
        "rate_by_category": {k: round(v, 4) for k, v in rate.items()},
        "category_spread": round(spread, 4),
        "category_oracle_auc": round(auc(y, orc), 3),
        "category_oracle_ci95": [round(lo, 3), round(hi, 3)],
        "transferred_110k_router_auc": round(auc(y, s_tr), 3),
        "prompt_length_auc": round(auc(y, L), 3),
        "loco": [{"category": c, "auc": round(a, 3)} for c, a in loco],
    }, indent=2) + "\n")
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
