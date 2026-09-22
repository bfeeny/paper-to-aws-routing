#!/usr/bin/env python3
"""Fetch graded benchmark items spanning easy to genuinely hard.

Routing labels are only meaningful if the prompt mix actually spans the tiers.
Grade-school arithmetic and knowledge recall will make any strong model look
pointless, because nothing in the set requires it. So the mix deliberately runs
from "a small model should handle this" to "this needs real reasoning":

    gsm8k       grade-school word problems        easy
    mmlu        knowledge recall, 4-way            easy-medium
    bbh         Big-Bench Hard reasoning tasks     medium-hard
    mmlu_pro    10-way, reasoning-heavy            hard
    math        competition maths, levels 4-5      hardest

Everything here grades exactly — number, letter, or normalized final answer — so
no judge is involved and none of its inconsistency enters the labels.

Uses the Hugging Face datasets-server JSON API: stdlib only, no parquet reader.

    python3 experiments/fetch_benchmarks.py --total 3000
"""

import argparse
import json
import pathlib
import random
import re
import urllib.parse
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments" / "benchmark_items.jsonl"
API = "https://datasets-server.huggingface.co/rows"

# Broad subject coverage, not just the quantitative ones.
MMLU_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge",
    "college_biology", "college_chemistry", "college_computer_science",
    "college_mathematics", "college_medicine", "college_physics", "computer_security",
    "econometrics", "electrical_engineering", "formal_logic", "global_facts",
    "high_school_european_history", "high_school_geography", "high_school_psychology",
    "high_school_statistics", "international_law", "jurisprudence", "machine_learning",
    "management", "marketing", "medical_genetics", "moral_disputes", "nutrition",
    "philosophy", "prehistory", "professional_accounting", "professional_law",
    "professional_medicine", "public_relations", "security_studies", "sociology",
    "virology", "world_religions",
]
MATH_SUBJECTS = ["algebra", "counting_and_probability", "geometry",
                 "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]
BBH_TASKS = [
    "causal_judgement", "date_understanding", "disambiguation_qa",
    "formal_fallacies", "geometric_shapes", "hyperbaton",
    "logical_deduction_five_objects", "movie_recommendation",
    "multistep_arithmetic_two", "navigate", "object_counting", "penguins_in_a_table",
    "reasoning_about_colored_objects", "ruin_names", "snarks", "temporal_sequences",
    "tracking_shuffled_objects_five_objects", "web_of_lies", "word_sorting",
]
LETTERS = "ABCDEFGHIJ"


def rows(dataset: str, config: str, split: str, offset: int, length: int):
    q = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": split,
                                "offset": offset, "length": length})
    try:
        with urllib.request.urlopen(f"{API}?{q}", timeout=120) as r:
            return json.load(r)["rows"]
    except Exception as e:  # noqa: BLE001 — a missing config shouldn't sink the run
        print(f"    ! {dataset}/{config}: {str(e)[:70]}")
        return []


def boxed(solution: str) -> str | None:
    """Pull the final answer out of \\boxed{...}, brace-balanced."""
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


def get_gsm8k(n):
    items, offset = [], 0
    while len(items) < n:
        batch = rows("openai/gsm8k", "main", "test", offset, min(100, n - len(items)))
        if not batch:
            break
        for row in batch:
            r = row["row"]
            items.append({
                "id": f"gsm8k-{row['row_idx']}", "benchmark": "gsm8k", "difficulty": "easy",
                "prompt": r["question"].strip(),
                "instruction": "Solve the problem. End your reply with 'Answer: <number>'.",
                "answer": r["answer"].split("####")[-1].strip().replace(",", ""),
                "grade": "number",
            })
        offset += len(batch)
    return items[:n]


def get_mmlu(n):
    per = max(1, n // len(MMLU_SUBJECTS))
    items = []
    for subj in MMLU_SUBJECTS:
        for row in rows("cais/mmlu", subj, "test", 0, min(100, per)):
            r = row["row"]
            body = "\n".join(f"{c}. {t}" for c, t in zip(LETTERS, r["choices"]))
            items.append({
                "id": f"mmlu-{subj}-{row['row_idx']}", "benchmark": "mmlu",
                "difficulty": "easy-medium", "subject": subj,
                "prompt": f"{r['question'].strip()}\n\n{body}",
                "instruction": "Answer with a single letter. End with 'Answer: <letter>'.",
                "answer": LETTERS[int(r["answer"])], "grade": "letter",
            })
    return items[:n]


def get_mmlu_pro(n):
    items, offset = [], 0
    while len(items) < n:
        batch = rows("TIGER-Lab/MMLU-Pro", "default", "test", offset, min(100, n - len(items)))
        if not batch:
            break
        for row in batch:
            r = row["row"]
            opts = r.get("options") or []
            body = "\n".join(f"{c}. {t}" for c, t in zip(LETTERS, opts))
            ans = r.get("answer") or (LETTERS[r["answer_index"]] if "answer_index" in r else None)
            if not ans:
                continue
            items.append({
                "id": f"mmlupro-{row['row_idx']}", "benchmark": "mmlu_pro",
                "difficulty": "hard", "subject": r.get("category"),
                "prompt": f"{r['question'].strip()}\n\n{body}",
                "instruction": "Answer with a single letter. End with 'Answer: <letter>'.",
                "answer": str(ans).strip().upper()[:1], "grade": "letter",
            })
        offset += len(batch)
    return items[:n]


def get_math(n):
    per = max(1, n // len(MATH_SUBJECTS))
    items = []
    for subj in MATH_SUBJECTS:
        offset = 0
        taken = 0
        while taken < per:
            batch = rows("EleutherAI/hendrycks_math", subj, "test", offset, 100)
            if not batch:
                break
            for row in batch:
                r = row["row"]
                level = str(r.get("level", ""))
                if not re.search(r"[45]", level):  # keep only the hard levels
                    continue
                ans = boxed(r.get("solution", ""))
                if not ans or len(ans) > 32:
                    continue
                items.append({
                    "id": f"math-{subj}-{row['row_idx']}", "benchmark": "math",
                    "difficulty": "hardest", "subject": subj, "level": level,
                    "prompt": r["problem"].strip(),
                    "instruction": "Solve. End your reply with 'Answer: <final answer>' "
                                   "using the simplest exact form.",
                    "answer": ans, "grade": "math",
                })
                taken += 1
                if taken >= per:
                    break
            offset += len(batch)
    return items[:n]


def get_bbh(n):
    per = max(1, n // len(BBH_TASKS))
    items = []
    for task in BBH_TASKS:
        for row in rows("lukaemon/bbh", task, "test", 0, min(100, per)):
            r = row["row"]
            items.append({
                "id": f"bbh-{task}-{row['row_idx']}", "benchmark": "bbh",
                "difficulty": "medium-hard", "subject": task,
                "prompt": r["input"].strip(),
                "instruction": "End your reply with 'Answer: <answer>' exactly as it "
                               "should appear.",
                "answer": str(r["target"]).strip(), "grade": "exact",
            })
    return items[:n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=20260921)
    args = ap.parse_args()

    # weighted toward the harder end: the point is to give the top tier something to do
    plan = [("gsm8k", get_gsm8k, 0.18), ("mmlu", get_mmlu, 0.22),
            ("bbh", get_bbh, 0.15), ("mmlu_pro", get_mmlu_pro, 0.27),
            ("math", get_math, 0.18)]

    items = []
    for name, fn, share in plan:
        want = int(args.total * share)
        print(f"  fetching {name} (target {want})")
        got = fn(want)
        print(f"    got {len(got)}")
        items += got

    random.Random(args.seed).shuffle(items)
    OUT.write_text("".join(json.dumps(i) + "\n" for i in items))

    counts, diff = {}, {}
    for i in items:
        counts[i["benchmark"]] = counts.get(i["benchmark"], 0) + 1
        diff[i["difficulty"]] = diff.get(i["difficulty"], 0) + 1
    print(f"\nwrote {len(items)} items to {OUT.relative_to(ROOT)}")
    print(f"  by benchmark: {counts}")
    print(f"  by difficulty: {diff}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
