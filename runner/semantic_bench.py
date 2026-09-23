#!/usr/bin/env python3
"""Does the semantic cache serve paraphrases without serving wrong answers?

Every triple in `results/reports/semantic-threshold-pairs.json` is a prompt, a
paraphrase of it (same answer) and a near-miss (one entity or number changed,
so a *different* answer). That is exactly the ground truth a semantic cache
needs: the paraphrase should be served from the cache, the near-miss must not
be, and the near-miss is on average the *closer* of the two in embedding space.

Each triple runs three calls against the live gateway:

  1. the original      -- expected miss, stores the answer and its vector
  2. the paraphrase    -- a hit is correct (a true hit)
  3. the near-miss     -- a hit is WRONG (a false hit: someone gets an answer
                          to a question they did not ask)

Run it twice, with `verify` on and off, to measure what the verifier buys:

    python3 runner/semantic_bench.py --n 20 --verify true
    python3 runner/semantic_bench.py --n 20 --verify false
"""
import argparse
import datetime as dt
import json
import pathlib
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gwclient import Client  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
PAIRS = ROOT / "results" / "reports" / "semantic-threshold-pairs.json"
MODEL = "mistral.ministral-3-3b-instruct"


def configure(session, param: str, verify: bool, threshold: float, ttl: float) -> dict:
    """Put the semantic cache in the chain right after the exact cache.

    The guardrail comes out for the duration: several of the benchmark prompts
    trip it, and a blocked request never reaches the cache, so leaving it in
    would measure the guardrail's coverage rather than the cache's. The config
    read here is returned so the caller can put it back.
    """
    ssm = session.client("ssm")
    before = ssm.get_parameter(Name=param)["Parameter"]["Value"]
    cfg = json.loads(before)
    plugins = [p for p in cfg["plugins"]
               if p["plugin"] not in ("semantic_cache", "guardrail")]
    entry = {"plugin": "semantic_cache",
             "params": {"threshold": threshold, "verify": verify, "top_k": 3, "ttl_s": 3600}}
    at = next((i for i, p in enumerate(plugins) if p["plugin"] == "cache"), 0) + 1
    plugins.insert(at, entry)
    for p in plugins:                       # ElastiCache is gone; DynamoDB is the store
        if p["plugin"] == "cache":
            p.setdefault("params", {})["backend"] = "dynamodb"
    cfg["plugins"] = plugins
    ssm.put_parameter(Name=param, Value=json.dumps(cfg), Overwrite=True, Type="String")
    print(f"semantic_cache verify={verify} threshold={threshold}, guardrail off; "
          f"waiting {ttl + 10:.0f}s for the config TTL")
    time.sleep(ttl + 10)
    return {"param": param, "value": before}


def ask(gw, text: str, tenant: str):
    status, _, raw, ms = gw.post(
        {"model": f"mantle/{MODEL}", "max_tokens": 80, "temperature": 0,
         "messages": [{"role": "user", "content": text}]},
        {"x-tenant-id": tenant})
    try:
        body = json.loads(raw)
    except Exception:  # noqa: BLE001
        body = {}
    x = body.get("x_gateway") or {}
    answer = ""
    if body.get("choices"):
        answer = (body["choices"][0].get("message") or {}).get("content") or ""
    err = "" if status == 200 else raw[:300].decode("utf-8", "replace")
    return {"status": status, "ms": round(ms, 1), "error": err, "cached": bool(x.get("cached")),
            "semantic": bool(x.get("semantic")), "similarity": x.get("similarity"),
            "answer": answer[:400]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", default="gwpipeline")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--verify", default="true", choices=["true", "false"])
    ap.add_argument("--threshold", type=float, default=0.80)
    ap.add_argument("--ttl", type=float, default=30)
    ap.add_argument("--skip-config", action="store_true")
    args = ap.parse_args()
    verify = args.verify == "true"

    gw = Client.for_stack(args.stack)
    saved = None
    if not args.skip_config:
        saved = configure(gw.session, gw.outputs["PipelineConfigParameter"],
                          verify, args.threshold, args.ttl)

    triples = json.loads(PAIRS.read_text())[: args.n]
    # A fresh tenant per run keeps the corpora separate: the vector index is
    # partitioned by tenant, so an earlier run cannot answer this one.
    tenant = f"sembench-{dt.datetime.now(dt.timezone.utc):%m%d%H%M}-{'v' if verify else 'nv'}"
    rows, started = [], dt.datetime.now(dt.timezone.utc)

    for i, t in enumerate(triples):
        seed = ask(gw, t["prompt"], tenant)
        time.sleep(1.0)                      # the vector is written on the way out
        para = ask(gw, t["paraphrase"], tenant)
        near = ask(gw, t["near_miss"], tenant)
        rows.append({"i": i, "seed": seed, "paraphrase": para, "near_miss": near,
                     "sim_paraphrase_offline": t["sim_paraphrase"],
                     "sim_near_miss_offline": t["sim_near_miss"]})
        if (i + 1) % 5 == 0:
            print(f"  {i + 1}/{len(triples)}")

    ok = [r for r in rows if r["seed"]["status"] == 200]
    true_hits = [r for r in ok if r["paraphrase"]["semantic"]]
    false_hits = [r for r in ok if r["near_miss"]["semantic"]]
    seed_ms = [r["seed"]["ms"] for r in ok if not r["seed"]["cached"]]
    hit_ms = [r["paraphrase"]["ms"] for r in true_hits]
    miss_ms = [r["near_miss"]["ms"] for r in ok if not r["near_miss"]["semantic"]]

    summary = {
        "verify": verify, "threshold": args.threshold, "tenant": tenant,
        "n": len(ok),
        "paraphrase_hit_rate": round(len(true_hits) / len(ok), 3) if ok else None,
        "near_miss_false_hit_rate": round(len(false_hits) / len(ok), 3) if ok else None,
        "median_ms": {
            "uncached_call": round(statistics.median(seed_ms), 1) if seed_ms else None,
            "semantic_hit": round(statistics.median(hit_ms), 1) if hit_ms else None,
            "semantic_miss": round(statistics.median(miss_ms), 1) if miss_ms else None,
        },
        "window": {"start": started.isoformat(),
                   "end": dt.datetime.now(dt.timezone.utc).isoformat()},
    }
    if saved:                                # put the chain back as it was found
        gw.session.client("ssm").put_parameter(Name=saved["param"], Value=saved["value"],
                                               Overwrite=True, Type="String")
    arm = f"{'verify' if verify else 'noverify'}-t{int(args.threshold * 100):03d}"
    out = ROOT / "results" / "reports" / f"semantic-cache-live-{arm}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1))
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
