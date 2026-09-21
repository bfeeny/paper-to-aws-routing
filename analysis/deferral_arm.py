#!/usr/bin/env python3
"""Escalate on the weak model's answer instead of predicting from the prompt.

Prompt-only routing has to know, before spending anything, whether a request
is hard. Every arm in this study says it cannot: ~0.55 on unseen groups
against a label ceiling of 0.912. Deferral asks a different and much easier
question -- here is an answer the cheap model already produced, does it look
like a failure? -- and answers it from metadata the response already carries.

Features, all free (no extra inference, nothing the gateway does not already
have in hand when the weak response returns):

    out_tokens       a struggling model rambles, restarts, second-guesses
    truncated        hit the token ceiling mid-reasoning
    no_answer_marker never reached a final answer
    n_markers        emitted several, i.e. changed its mind
    backtracking     "wait", "actually", "let me reconsider", "recount"
    hedging          "approximately", "I think", "not sure"

The comparison that matters is not against zero. It is against the prompt-only
router evaluated the same way -- same split, same confound suite -- because the
whole claim is that the cheap post-hoc signal beats the expensive a-priori one.

    .venv/bin/python analysis/deferral_arm.py --labels experiments/math_labels.jsonl \
        --items experiments/math_items.jsonl --group subject
"""
import argparse, json, pathlib, re, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cascade_sweep import auc, auc_ci
from train_router import embed, fit_logistic, scores

ROOT = pathlib.Path(__file__).resolve().parent.parent
BACKTRACK = re.compile(r"\b(wait|actually|hold on|let me (?:re|double)|recount|recheck|"
                       r"reconsider|on second thought|i made (?:a|an) (?:mistake|error))\b", re.I)
HEDGE = re.compile(r"\b(approximately|roughly|i think|not (?:entirely )?sure|"
                   r"probably|it seems|might be|unclear)\b", re.I)


def features(rec: dict) -> list[float]:
    t = rec["tiers"].get("haiku", {}) or {}
    tail = t.get("reply_tail") or ""
    n_mark = len(re.findall(r"[Aa]nswer\s*:", tail))
    return [
        float(t.get("out") or 0),
        1.0 if t.get("truncated") else 0.0,
        0.0 if n_mark else 1.0,
        float(min(n_mark, 4)),
        float(len(BACKTRACK.findall(tail))),
        float(len(HEDGE.findall(tail))),
        float(len(tail)),
    ]


NAMES = ["out_tokens", "truncated", "no_marker", "n_markers", "backtrack", "hedge", "chars"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="experiments/math_labels.jsonl")
    ap.add_argument("--items", default="experiments/math_items.jsonl")
    ap.add_argument("--group", default="subject")
    ap.add_argument("--min-pos", type=int, default=5)
    ap.add_argument("--out", default="results/reports/deferral")
    args = ap.parse_args()

    recs = [json.loads(l) for l in (ROOT / args.labels).read_text().splitlines() if l.strip()]
    items = {json.loads(l)["id"]: json.loads(l)
             for l in (ROOT / args.items).read_text().splitlines() if l.strip()}
    recs = [r for r in recs if r["id"] in items]
    g = np.array([str(items[r["id"]].get(args.group)) for r in recs])
    y = np.isin([r["tier_needed"] for r in recs], [1, 2]).astype(float)
    F = np.array([features(r) for r in recs], dtype=float)
    # standardise; the raw scales differ by three orders of magnitude
    F = (F - F.mean(0)) / np.where(F.std(0) > 0, F.std(0), 1.0)
    P = embed([r["prompt"] for r in recs], "personal", "us-east-1", 256, workers=24)
    print(f"n={len(recs)} groups={len(set(g))} escalation={y.mean():.1%}")

    rng = np.random.default_rng(20260921)
    perm = rng.permutation(len(recs)); cut = int(len(recs) * 0.70)
    tr, te = perm[:cut], perm[cut:]

    def fit_score(X, tr_i, te_i, yy):
        w, b = fit_logistic(X[tr_i], yy[tr_i], l2=1.0)
        return scores(X[te_i], w, b)

    rows = []

    def add(name, yy, ss, note=""):
        lo, hi = auc_ci(yy, ss)
        print(f"  {name:38} AUC {auc(yy, ss):.3f}  [{lo:.3f}, {hi:.3f}] {note}")
        rows.append({"arm": name, "auc": round(auc(yy, ss), 3),
                     "ci95": [round(lo, 3), round(hi, 3)], "note": note})

    print("\nrandom split:")
    add("deferral (response features)", y[te], fit_score(F, tr, te, y))
    add("  out_tokens alone", y[te], F[te, 0])
    add("prompt-only router (Titan)", y[te], fit_score(P, tr, te, y))

    # the diagnostic that killed every previous arm
    ys = y.copy()
    for grp in set(g):
        m = np.where(g == grp)[0]
        ys[m] = rng.permutation(y[m])
    real = auc(y[te], fit_score(F, tr, te, y))
    shuf = auc(y[te], fit_score(F, tr, te, ys))
    print(f"\nwithin-group label shuffle: {shuf:.3f} vs real {real:.3f} "
          f"-> {max(shuf,0)/real:.0%} is group lookup")

    print("\nleave-one-group-out:")
    d_l, p_l, t_l = [], [], []
    for grp in sorted(set(g)):
        m = g == grp
        if y[m].sum() < args.min_pos or (1 - y[m]).sum() < args.min_pos:
            continue
        tr_i, te_i = np.where(~m)[0], np.where(m)[0]
        d = auc(y[m], fit_score(F, tr_i, te_i, y))
        p = auc(y[m], fit_score(P, tr_i, te_i, y))
        t = auc(y[m], F[m, 0])
        d_l.append(d); p_l.append(p); t_l.append(t)
        print(f"  {grp:30} deferral {d:.3f}   prompt {p:.3f}   out_tokens {t:.3f}")
    print(f"  {'MEAN':30} deferral {np.mean(d_l):.3f}   prompt {np.mean(p_l):.3f}"
          f"   out_tokens {np.mean(t_l):.3f}")

    out = ROOT / f"{args.out}-{pathlib.Path(args.labels).stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "labels": args.labels, "n": len(recs), "escalation_rate": round(float(y.mean()), 4),
        "random_split": rows, "within_group_shuffle": {"real": round(real, 3),
                                                       "shuffled": round(shuf, 3)},
        "loto_mean": {"deferral": round(float(np.mean(d_l)), 3),
                      "prompt_only": round(float(np.mean(p_l)), 3),
                      "out_tokens_alone": round(float(np.mean(t_l)), 3)},
    }, indent=2) + "\n")
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
