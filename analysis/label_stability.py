#!/usr/bin/env python3
"""How reliable is a cascade label? Resample each tier and measure.

Every AUC in this study is bounded by the reliability of its target. The
cascade label is three single draws at temperature 0, and Bedrock at
temperature 0 is not deterministic, so a borderline item -- exactly the kind a
router must rank -- can land on either side of its tier boundary by chance.
Nothing so far measures that, which means no reported AUC can be read against
the ceiling the labels permit.

Two temperatures, because they answer different questions:

  T=0  test-retest of the labelling procedure as actually run. Disagreement
       here is pure instrument noise.
  T=1  the model's own answer distribution, which gives a soft label
       P(correct) instead of a coin flip, and an attenuation ceiling.

    python3 analysis/label_stability.py --items 200 --samples 4
"""

import argparse
import collections
import concurrent.futures as cf
import json
import pathlib
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import botocore.session
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from grading import graded  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
REGION = "us-east-1"
TIERS = [
    ("haiku", "us.anthropic.claude-haiku-4-5-20251001-v1:0"),
    ("sonnet", "us.anthropic.claude-sonnet-4-6"),
    ("opus", "us.anthropic.claude-opus-4-5-20251101-v1:0"),
]
SOURCES = [("experiments/cascade_labels.jsonl", "experiments/benchmark_items.jsonl"),
           ("experiments/bbh_labels.jsonl", "experiments/bbh_items.jsonl"),
           ("experiments/math_labels.jsonl", "experiments/math_items.jsonl")]


def converse(creds, model_id, prompt, max_tokens, temperature, attempts=4):
    url = (f"https://bedrock-runtime.{REGION}.amazonaws.com/model/"
           f"{urllib.parse.quote(model_id, safe='')}/converse")
    body = json.dumps({
        "messages": [{"role": "user", "content": [{"text": prompt}]}],
        "inferenceConfig": {"maxTokens": max_tokens, "temperature": temperature},
    }).encode()
    for attempt in range(attempts):
        req = AWSRequest(method="POST", url=url, data=body,
                         headers={"Content-Type": "application/json"})
        SigV4Auth(creds, "bedrock", REGION).add_auth(req)
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, data=body, headers=dict(req.headers)),
                timeout=240,
            ) as r:
                d = json.load(r)
                return "".join(c.get("text", "") for c in d["output"]["message"]["content"])
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and attempt < attempts - 1:
                time.sleep(2 * (attempt + 1))
                continue
            return None
        except Exception:  # noqa: BLE001
            if attempt < attempts - 1:
                time.sleep(1.5 * (attempt + 1))
                continue
            return None
    return None


def one(creds, item, tier_name, model_id, temp, max_tokens):
    prompt = f"{item['prompt']}\n\n{item['instruction']}"
    reply = converse(creds, model_id, prompt, max_tokens, temp)
    return {"id": item["id"], "tier": tier_name, "temp": temp,
            "correct": graded(reply, item)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=int, default=200)
    ap.add_argument("--samples", type=int, default=4, help="draws per tier per temperature")
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--profile", default="personal")
    ap.add_argument("--out", default="experiments/stability_raw.jsonl")
    args = ap.parse_args()

    # stratify by (benchmark, labelled tier) so every cell of the label is tested
    items, labels = {}, {}
    for lab_f, item_f in SOURCES:
        lp, ip = ROOT / lab_f, ROOT / item_f
        if not (lp.exists() and ip.exists()):
            continue
        for l in ip.read_text().splitlines():
            if l.strip():
                d = json.loads(l)
                items[d["id"]] = d
        for l in lp.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                if r["id"] in items:
                    labels[r["id"]] = r

    strata = collections.defaultdict(list)
    for rid, r in labels.items():
        strata[(r["benchmark"], r["tier_needed"])].append(rid)
    rng = random.Random(args.seed)
    per = max(1, args.items // max(len(strata), 1))
    chosen = []
    for key, ids in sorted(strata.items()):
        chosen += rng.sample(ids, min(per, len(ids)))
    chosen = chosen[: args.items]
    print(f"{len(chosen)} items across {len(strata)} (benchmark, tier) strata")
    print(f"calls: {len(chosen)} x {len(TIERS)} tiers x {args.samples * 2} draws "
          f"= {len(chosen) * len(TIERS) * args.samples * 2}")

    creds = botocore.session.Session(profile=args.profile).get_credentials().get_frozen_credentials()
    jobs = [(items[rid], nm, mid, t)
            for rid in chosen for nm, mid in TIERS
            for t in (0.0, 1.0) for _ in range(args.samples)]

    out_path = ROOT / args.out
    done = 0
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex, out_path.open("w") as fh:
        futs = [ex.submit(one, creds, it, nm, mid, t, args.max_tokens)
                for it, nm, mid, t in jobs]
        for fut in cf.as_completed(futs):
            fh.write(json.dumps(fut.result()) + "\n")
            done += 1
            if done % 250 == 0:
                print(f"  {done}/{len(jobs)}", flush=True)
    print(f"wrote {out_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
