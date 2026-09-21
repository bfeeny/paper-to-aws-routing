#!/usr/bin/env python3
"""Re-grade stored cascade replies with the current grader. No inference.

This is why label_cascade.py keeps `reply_tail`: a grader defect found after
the fact costs a few seconds of CPU instead of thousands of dollars of
re-labelling. Truncated replies are recorded as `truncated` rather than wrong,
because "ran out of tokens" is missing data, not a capability ceiling.

    python3 analysis/regrade_offline.py experiments/bbh_labels.jsonl
"""
import collections
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from grading import graded  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
TIERS = ["haiku", "sonnet", "opus"]


def regrade(path: pathlib.Path, items: dict, cap: int) -> dict:
    recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    flips = collections.Counter()
    before = collections.Counter(r["tier_needed"] for r in recs)
    for r in recs:
        item = items.get(r["id"])
        if not item:
            continue
        needed = len(TIERS)
        for idx, name in enumerate(TIERS):
            t = r["tiers"].get(name)
            if t is None:
                continue
            was = bool(t.get("correct"))
            now = graded(t.get("reply_tail"), item)
            t["correct"] = now
            t["truncated"] = (t.get("out") or 0) >= cap - 4
            if now and not was:
                flips[name] += 1
            if now:
                needed = idx
                break
        r["tier_needed"] = needed
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))
    after = collections.Counter(r["tier_needed"] for r in recs)
    return {"n": len(recs), "rescued": dict(flips), "before": before, "after": after}


def main() -> int:
    targets = sys.argv[1:] or ["experiments/cascade_labels.jsonl",
                               "experiments/bbh_labels.jsonl",
                               "experiments/math_labels.jsonl"]
    items = {}
    for f in ("benchmark_items", "bbh_items", "math_items"):
        p = ROOT / f"experiments/{f}.jsonl"
        if p.exists():
            for l in p.read_text().splitlines():
                if l.strip():
                    d = json.loads(l)
                    items[d["id"]] = d
    names = TIERS + ["none"]
    for t in targets:
        p = ROOT / t
        if not p.exists():
            print(f"skip {t} (missing)")
            continue
        cap = 2048 if "math" in t else 1024
        st = regrade(p, items, cap)
        esc_b = (st["before"][1] + st["before"][2]) / st["n"]
        esc_a = (st["after"][1] + st["after"][2]) / st["n"]
        print(f"\n{t}  n={st['n']}  rescued={st['rescued']}")
        print(f"  {'tier':8}{'before':>9}{'after':>9}")
        for i, nm in enumerate(names):
            print(f"  {nm:8}{st['before'][i]/st['n']:8.1%}{st['after'][i]/st['n']:9.1%}")
        print(f"  escalation-worthy: {esc_b:.1%} -> {esc_a:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
