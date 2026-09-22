#!/usr/bin/env python3
"""Fetch MATH across all levels as a single-task difficulty test.

BBH failed the held-out-task split because each of its 27 tasks has a visually
distinct template, so a model can name the task and look up its escalation
rate without ever modeling difficulty. MATH removes that affordance: every
item is "solve this competition problem", one template, and the difficulty is
latent in the problem rather than announced by its format.

It also ships ground-truth difficulty (levels 1-5), so we can ask directly
whether the router's score tracks the label a human assigned.

    python3 experiments/fetch_math.py --per-subject 360
"""

import argparse
import json
import pathlib
import random
import re
import time
import urllib.parse
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments" / "math_items.jsonl"
API = "https://datasets-server.huggingface.co/rows"
SUBJECTS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
            "number_theory", "prealgebra", "precalculus"]


def rows(config, offset, length, attempts=5):
    q = urllib.parse.urlencode({"dataset": "EleutherAI/hendrycks_math", "config": config,
                                "split": "test", "offset": offset, "length": length})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(f"{API}?{q}", timeout=120) as r:
                return json.load(r)["rows"]
        except Exception as e:  # noqa: BLE001
            if attempt == attempts - 1:
                print(f"    ! {config}: {str(e)[:60]}")
                return []
            time.sleep(2 * (attempt + 1))
    return []


def boxed(solution: str):
    i = solution.rfind("\\boxed")
    if i < 0:
        return None
    j = solution.find("{", i)
    if j < 0:
        return None
    depth, out = 0, []
    for ch in solution[j:]:
        if ch == "{":
            depth += 1
            if depth == 1:
                continue
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(ch)
    return "".join(out).strip() or None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-subject", type=int, default=360)
    ap.add_argument("--seed", type=int, default=20260921)
    args = ap.parse_args()

    items = []
    for subj in SUBJECTS:
        got, offset = 0, 0
        while got < args.per_subject:
            batch = rows(subj, offset, 100)
            if not batch:
                break
            for row in batch:
                r = row["row"]
                ans = boxed(r.get("solution", ""))
                if not ans or len(ans) > 32:
                    continue
                lvl = re.search(r"(\d)", str(r.get("level", "")))
                items.append({
                    "id": f"math-{subj}-{row['row_idx']}", "benchmark": "math",
                    "subject": subj, "level": int(lvl.group(1)) if lvl else None,
                    "prompt": r["problem"].strip(),
                    "instruction": "Solve. End your reply with 'Answer: <final answer>' "
                                   "using the simplest exact form.",
                    "answer": ans, "grade": "math",
                })
                got += 1
                if got >= args.per_subject:
                    break
            offset += len(batch)
        print(f"  {subj:28} {got:4}")

    random.Random(args.seed).shuffle(items)
    OUT.write_text("".join(json.dumps(i) + "\n" for i in items))
    from collections import Counter
    print(f"\nwrote {len(items)} items to {OUT.relative_to(ROOT)}")
    print("  by level:", dict(sorted(Counter(i["level"] for i in items).items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
