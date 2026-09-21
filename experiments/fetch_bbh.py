#!/usr/bin/env python3
"""Fetch the full Big-Bench Hard suite as a single-distribution test set.

The mixed benchmark set produced an AUC that turned out to be mostly
benchmark-style detection: escalation rates run from 1.9% on GSM8K to 25.2% on
BBH, so a model that merely recognises which corpus a prompt came from scores
well without knowing anything about difficulty. A deployment sees one traffic
distribution, so the honest test is within a single corpus.

BBH is the right one to scale up: it has the highest escalation rate of the
five, all 27 tasks grade exactly, and the tasks are heterogeneous enough that
"one distribution" is not a euphemism for "one template".

    python3 experiments/fetch_bbh.py --per-task 90
"""

import argparse
import json
import pathlib
import random
import time
import urllib.parse
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments" / "bbh_items.jsonl"
API = "https://datasets-server.huggingface.co/rows"

TASKS = [
    "boolean_expressions", "causal_judgement", "date_understanding",
    "disambiguation_qa", "dyck_languages", "formal_fallacies", "geometric_shapes",
    "hyperbaton", "logical_deduction_five_objects", "logical_deduction_seven_objects",
    "logical_deduction_three_objects", "movie_recommendation", "multistep_arithmetic_two",
    "navigate", "object_counting", "penguins_in_a_table",
    "reasoning_about_colored_objects", "ruin_names",
    "salient_translation_error_detection", "snarks", "sports_understanding",
    "temporal_sequences", "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects", "tracking_shuffled_objects_three_objects",
    "web_of_lies", "word_sorting",
]


def rows(config: str, offset: int, length: int, attempts: int = 5):
    q = urllib.parse.urlencode({"dataset": "lukaemon/bbh", "config": config,
                                "split": "test", "offset": offset, "length": length})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(f"{API}?{q}", timeout=120) as r:
                return json.load(r)["rows"]
        except Exception as e:  # noqa: BLE001 — the server rate-limits freely
            if attempt == attempts - 1:
                print(f"    ! {config}: {str(e)[:70]}")
                return []
            time.sleep(2 * (attempt + 1))  # 429s are the common case; back off
    return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-task", type=int, default=90)
    ap.add_argument("--seed", type=int, default=20260921)
    args = ap.parse_args()

    items = []
    for task in TASKS:
        got, offset = 0, 0
        while got < args.per_task:
            batch = rows(task, offset, min(100, args.per_task - got))
            if not batch:
                break
            for row in batch:
                r = row["row"]
                items.append({
                    "id": f"bbh-{task}-{row['row_idx']}", "benchmark": "bbh",
                    "difficulty": "medium-hard", "subject": task,
                    "prompt": r["input"].strip(),
                    "instruction": "End your reply with 'Answer: <answer>' exactly as it "
                                   "should appear.",
                    "answer": str(r["target"]).strip(), "grade": "exact",
                })
            got += len(batch)
            offset += len(batch)
        print(f"  {task:44} {got:4}")

    random.Random(args.seed).shuffle(items)
    OUT.write_text("".join(json.dumps(i) + "\n" for i in items))
    print(f"\nwrote {len(items)} items to {OUT.relative_to(ROOT)} across {len(TASKS)} tasks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
