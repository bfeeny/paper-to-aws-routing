#!/usr/bin/env python3
"""Label benchmark items by the cheapest Claude tier that answers them correctly.

    Haiku   correct -> tier 0   the cheap model suffices
    Sonnet  correct -> tier 1   worth escalating
    Opus    correct -> tier 2   only the top tier gets it
    none    correct -> tier 3   nobody gets it; routing up buys nothing

This is the ordinal generalisation of RouteLLM's binary label, and it is what a
three-tier router actually needs. Each tier is called only when the cheaper one
failed, so cost scales with difficulty rather than with the number of tiers.

Grading is exact (GSM8K: final number; MMLU: letter), so no judge is involved and
none of its inconsistency enters the labels.

    python3 analysis/label_cascade.py --limit 100
"""

import argparse
import concurrent.futures as cf
import collections
import json
import pathlib
import re
import time
import urllib.error
import urllib.parse
import urllib.request

import botocore.session
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

ROOT = pathlib.Path(__file__).resolve().parent.parent
REGION = "us-east-1"
TIERS = [
    ("haiku", "us.anthropic.claude-haiku-4-5-20251001-v1:0"),
    ("sonnet", "us.anthropic.claude-sonnet-4-6"),
    ("opus", "us.anthropic.claude-opus-4-5-20251101-v1:0"),
]


def converse(creds, model_id: str, prompt: str, max_tokens: int, attempts: int = 4):
    url = (f"https://bedrock-runtime.{REGION}.amazonaws.com/model/"
           f"{urllib.parse.quote(model_id, safe='')}/converse")
    body = json.dumps({
        "messages": [{"role": "user", "content": [{"text": prompt}]}],
        "inferenceConfig": {"maxTokens": max_tokens, "temperature": 0},
    }).encode()
    for attempt in range(attempts):
        req = AWSRequest(method="POST", url=url, data=body,
                         headers={"Content-Type": "application/json"})
        SigV4Auth(creds, "bedrock", REGION).add_auth(req)
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, data=body, headers=dict(req.headers)),
                timeout=180,
            ) as r:
                d = json.load(r)
                text = "".join(c.get("text", "")
                               for c in d["output"]["message"]["content"])
                return text, d.get("usage", {}), (time.perf_counter() - started) * 1000
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and attempt < attempts - 1:
                time.sleep(2 * (attempt + 1))
                continue
            return None, {"error": e.read().decode()[:160]}, 0.0
    return None, {"error": "retries exhausted"}, 0.0


def _norm_math(s: str) -> str:
    """Normalise a LaTeX-ish final answer enough to compare two spellings of it.

    Deliberately conservative: it collapses formatting the model chooses freely
    (\\left, \\!, $, whitespace, a trailing period) and nothing that could change
    the value. Anything it cannot normalise stays unequal rather than guessing.
    """
    s = s.strip().strip("$").strip()
    s = re.sub(r"\\(?:left|right|!|,|;|:|\s)", "", s)
    s = re.sub(r"\\d?frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", s)
    s = re.sub(r"\\text\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\s+", "", s)
    s = s.replace("^{\\circ}", "").replace("^\\circ", "").replace("\\%", "")
    s = s.rstrip(".")
    return s.lower()


def graded(reply: str | None, item: dict) -> bool:
    """Grade a reply against the item's declared answer format.

    Each benchmark carries its own `grade` kind, and they genuinely differ:
    MMLU is 4-way but MMLU-Pro is 10-way, BBH answers are free-form strings, and
    MATH answers are expressions. Grading them all as numbers — which an earlier
    version did — silently marks every non-numeric answer wrong, which shows up
    as a spuriously high "no tier could answer" rate rather than as an error.
    """
    if not reply:
        return False
    tail = reply.strip()
    kind = item.get("grade")
    want = str(item["answer"]).strip()

    if kind == "letter":
        # A-J: MMLU-Pro has ten options, not four.
        m = re.findall(r"Answer:\s*\(?([A-J])\)?", tail, re.I) or \
            re.findall(r"\b([A-J])\b", tail)
        return bool(m) and m[-1].upper() == want.upper()

    if kind == "number":
        m = re.findall(r"Answer:\s*\$?(-?[\d,]+(?:\.\d+)?)", tail, re.I) or \
            re.findall(r"(-?[\d,]+(?:\.\d+)?)", tail)
        if not m:
            return False
        try:
            return abs(float(m[-1].replace(",", "")) - float(want)) < 1e-6
        except ValueError:
            return False

    # "exact" (BBH) and "math": compare the tail after the Answer: marker.
    m = re.findall(r"Answer:\s*(.+)", tail, re.I)
    got = m[-1].strip() if m else tail.splitlines()[-1].strip()

    if kind == "math":
        if _norm_math(got) == _norm_math(want):
            return True
        try:  # the same value spelled 0.5 and \frac{1}{2} should still match
            return abs(float(got.strip("$")) - float(want)) < 1e-6
        except ValueError:
            return False

    # exact: case- and punctuation-insensitive, tolerating "(A)" vs "A"
    def norm(s: str) -> str:
        return re.sub(r"[\s().,]", "", s).lower()

    return norm(got) == norm(want)


def label_item(creds, item: dict, max_tokens: int) -> dict:
    prompt = f"{item['prompt']}\n\n{item['instruction']}"
    rec = {"id": item["id"], "benchmark": item["benchmark"], "prompt": item["prompt"],
           "tiers": {}, "tier_needed": len(TIERS)}
    for idx, (name, model_id) in enumerate(TIERS):
        reply, usage, ms = converse(creds, model_id, prompt, max_tokens)
        ok = graded(reply, item)
        rec["tiers"][name] = {
            "correct": ok,
            "in": usage.get("inputTokens"), "out": usage.get("outputTokens"),
            "latency_ms": round(ms, 1),
            "error": usage.get("error"),
            # Keep the tail of every reply. Grading only ever reads the end, and
            # storing it means a grader fix can be replayed offline instead of
            # re-paying for 5,000 inference calls to find out what changed.
            "reply_tail": (reply or "")[-800:],
        }
        if ok:
            rec["tier_needed"] = idx
            break
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="experiments/benchmark_items.jsonl")
    ap.add_argument("--out", default="experiments/cascade_labels.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--profile", default="personal")
    args = ap.parse_args()

    items = [json.loads(l) for l in (ROOT / args.items).read_text().splitlines() if l.strip()]
    if args.limit:
        items = items[: args.limit]
    creds = botocore.session.Session(profile=args.profile).get_credentials().get_frozen_credentials()

    print(f"labelling {len(items)} items through {len(TIERS)} tiers (cheapest first)")
    out_path = ROOT / args.out
    recs = []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex, out_path.open("w") as fh:
        futures = [ex.submit(label_item, creds, it, args.max_tokens) for it in items]
        for i, fut in enumerate(cf.as_completed(futures), 1):
            rec = fut.result()
            recs.append(rec)
            fh.write(json.dumps(rec) + "\n")
            if i % 25 == 0:
                print(f"  {i}/{len(items)}")

    dist = collections.Counter(r["tier_needed"] for r in recs)
    names = [n for n, _ in TIERS] + ["none"]
    tokens = collections.defaultdict(lambda: [0, 0, 0])
    for r in recs:
        for name, t in r["tiers"].items():
            tokens[name][0] += 1
            tokens[name][1] += t.get("in") or 0
            tokens[name][2] += t.get("out") or 0

    print("\ncheapest tier that answered correctly:")
    for idx, name in enumerate(names):
        n = dist.get(idx, 0)
        print(f"  {name:6} {n:4}  ({n / max(len(recs), 1):.1%})")
    print("\nmodel usage (what the labelling itself cost):")
    for name, (calls, tin, tout) in tokens.items():
        print(f"  {name:6} {calls:4} calls  in={tin:7}  out={tout:7}")
    errs = sum(1 for r in recs for t in r["tiers"].values() if t.get("error"))
    if errs:
        print(f"\n  {errs} tier calls errored (recorded, not dropped)")
    print(f"\nwrote {out_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
