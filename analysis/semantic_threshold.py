#!/usr/bin/env python3
"""Is there a safe similarity threshold for a semantic cache?

A semantic cache serves a stored answer when a new prompt is "close enough" to
an old one. That is only sound if paraphrases of the same question sit clearly
above the threshold while questions with different answers sit clearly below
it. The dangerous case is not a random pair; it is a near-identical prompt with
a different answer -- one changed number, one swapped entity.

Builds three sets with Claude, embeds everything with Titan, and reports the
similarity distributions and what any threshold would cost in false hits.

    python3 analysis/semantic_threshold.py --n 40
"""
import argparse
import concurrent.futures as cf
import json
import pathlib
import re
import statistics
import sys
import urllib.parse
import urllib.request

import botocore.session
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

ROOT = pathlib.Path(__file__).resolve().parent.parent
REGION = "us-east-1"
WRITER = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
EMBED = "amazon.titan-embed-text-v2:0"


def _creds(profile):
    return botocore.session.Session(profile=profile).get_credentials().get_frozen_credentials()


def _post(creds, url, body, service="bedrock"):
    data = json.dumps(body).encode()
    req = AWSRequest(method="POST", url=url, data=data, headers={"Content-Type": "application/json"})
    SigV4Auth(creds, service, REGION).add_auth(req)
    with urllib.request.urlopen(
        urllib.request.Request(url, data=data, headers=dict(req.headers)), timeout=120
    ) as r:
        return json.load(r)


def rewrite(creds, prompt: str, kind: str) -> str | None:
    instruction = {
        "paraphrase": ("Rewrite the question so it means exactly the same thing and has the same "
                       "answer. Change the wording, not the meaning. Reply with the rewritten "
                       "question only."),
        "near_miss": ("Change the question as little as possible so that it now has a DIFFERENT "
                      "correct answer -- swap one number, name, date or entity. Keep the wording "
                      "otherwise identical. Reply with the changed question only."),
    }[kind]
    url = (f"https://bedrock-runtime.{REGION}.amazonaws.com/model/"
           f"{urllib.parse.quote(WRITER, safe='')}/converse")
    try:
        d = _post(creds, url, {"messages": [{"role": "user", "content": [
            {"text": f"{instruction}\n\nQuestion: {prompt}"}]}],
            "inferenceConfig": {"maxTokens": 300, "temperature": 0}})
        return "".join(c.get("text", "") for c in d["output"]["message"]["content"]).strip()
    except Exception:  # noqa: BLE001
        return None


def embed(creds, text: str, dims: int = 1024):
    url = f"https://bedrock-runtime.{REGION}.amazonaws.com/model/{EMBED}/invoke"
    d = _post(creds, url, {"inputText": text[:8000], "dimensions": dims, "normalize": True})
    return d["embedding"]


def cos(a, b):
    return sum(x * y for x, y in zip(a, b))       # both are unit vectors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--dims", type=int, default=1024, choices=[256, 512, 1024],
                    help="Titan V2 is Matryoshka: 256 is a truncation of the native 1024")
    ap.add_argument("--profile", default="personal")
    args = ap.parse_args()
    creds = _creds(args.profile)

    src = ROOT / "experiments" / "prompts" / "mt_bench.jsonl"
    prompts = [json.loads(l)["prompt"] for l in src.read_text().splitlines() if l.strip()][: args.n]
    print(f"{len(prompts)} base prompts")

    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        paras = list(ex.map(lambda p: rewrite(creds, p, "paraphrase"), prompts))
        nears = list(ex.map(lambda p: rewrite(creds, p, "near_miss"), prompts))
        texts = prompts + [p or "" for p in paras] + [n or "" for n in nears]
        vecs = list(ex.map(lambda t: embed(creds, t, args.dims) if t else None, texts))

    n = len(prompts)
    base, para_v, near_v = vecs[:n], vecs[n:2 * n], vecs[2 * n:]
    same = [cos(base[i], para_v[i]) for i in range(n) if para_v[i]]
    diff = [cos(base[i], near_v[i]) for i in range(n) if near_v[i]]
    rand = [cos(base[i], base[j]) for i in range(n) for j in range(n) if i < j]

    def band(name, xs):
        xs = sorted(xs)
        print(f"  {name:22} n={len(xs):4}  min {xs[0]:.3f}  p10 {xs[len(xs)//10]:.3f}  "
              f"median {statistics.median(xs):.3f}  p90 {xs[int(len(xs)*0.9)]:.3f}  max {xs[-1]:.3f}")
        return xs

    print("\ncosine similarity to the original prompt:")
    same_s = band("paraphrase (should hit)", same)
    diff_s = band("near miss (must not)", diff)
    band("unrelated prompt", rand)

    print("\nthreshold trade-off:")
    for th in (0.80, 0.85, 0.90, 0.92, 0.95, 0.97, 0.99):
        hits = sum(s >= th for s in same_s) / len(same_s)
        false = sum(s >= th for s in diff_s) / len(diff_s)
        print(f"  >= {th:.2f}   serves {hits:5.0%} of paraphrases   "
              f"and {false:5.0%} of different-answer prompts")

    def flat(s):
        return re.sub(r"\s+", " ", s or "")[:70]

    worst = sorted(range(n), key=lambda i: -(cos(base[i], near_v[i]) if near_v[i] else -1))[:3]
    print("\nclosest near misses:")
    for i in worst:
        if near_v[i]:
            sim = cos(base[i], near_v[i])
            print(f"  {sim:.3f}  {flat(prompts[i])!r}")
            print(f"         vs {flat(nears[i])!r}")
    # A guard: paraphrases keep the facts that decide the answer; a near miss
    # changes one. Require the prompt's numbers and capitalised words to match
    # before a semantic hit is allowed.
    def facts(s: str):
        nums = tuple(sorted(re.findall(r"\d+(?:\.\d+)?", s or "")))
        caps = tuple(sorted({w for w in re.findall(r"\b[A-Z][a-z]{2,}\b", s or "")}))
        return nums, caps

    def guarded(th: float):
        hit = sum(1 for i in range(n) if para_v[i] and cos(base[i], para_v[i]) >= th
                  and facts(prompts[i]) == facts(paras[i]))
        bad = sum(1 for i in range(n) if near_v[i] and cos(base[i], near_v[i]) >= th
                  and facts(prompts[i]) == facts(nears[i]))
        return hit / n, bad / n

    print("\nwith a numbers-and-names guard:")
    for th in (0.80, 0.85, 0.90, 0.95):
        h, b = guarded(th)
        print(f"  >= {th:.2f}   serves {h:5.0%} of paraphrases   and {b:5.0%} of different-answer prompts")

    (ROOT / "results" / "reports" / f"semantic-threshold-pairs-{args.dims}.json").write_text(json.dumps(
        [{"prompt": prompts[i], "paraphrase": paras[i], "near_miss": nears[i],
          "sim_paraphrase": cos(base[i], para_v[i]) if para_v[i] else None,
          "sim_near_miss": cos(base[i], near_v[i]) if near_v[i] else None}
         for i in range(n)], indent=2) + "\n")
    out = ROOT / "results" / "reports" / f"semantic-threshold-{args.dims}.json"
    out.write_text(json.dumps({
        "n": n, "dims": args.dims,
        "paraphrase": {"median": statistics.median(same_s), "p10": same_s[len(same_s)//10], "min": same_s[0]},
        "near_miss": {"median": statistics.median(diff_s), "p90": diff_s[int(len(diff_s)*0.9)], "max": diff_s[-1]},
        "guarded": {str(th): dict(zip(("paraphrase_hit_rate", "false_hit_rate"), guarded(th)))
                    for th in (0.80, 0.85, 0.90, 0.95)},
        "thresholds": {str(th): {"paraphrase_hit_rate": sum(s >= th for s in same_s) / len(same_s),
                                 "false_hit_rate": sum(s >= th for s in diff_s) / len(diff_s)}
                       for th in (0.80, 0.85, 0.90, 0.92, 0.95, 0.97, 0.99)},
    }, indent=2) + "\n")
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
