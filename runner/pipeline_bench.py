#!/usr/bin/env python3
"""Measure what each layer of gateway customization adds to request latency.

Paired design: every gateway call is preceded by a direct call to Bedrock
(bedrock-mantle) with the same model and prompt. Model latency drifts by
hundreds of milliseconds over minutes; pairing each gateway call with a direct
call made a moment earlier cancels most of that drift, so the per-pair
difference estimates the overhead of the path rather than the model's mood.

Conditions (one per run):

  none           gateway with no interceptor attached (deploy INTERCEPTOR=false)
  empty          interceptor on both phases, empty plugin chain
  no_guardrail   the full chain minus the guardrail plugin
  full           tenant, max_tokens, router, model_policy, budget, guardrail, metering

The chain conditions rewrite the SSM parameter and wait out the config TTL, so
all three run against one deployment. Gateway calls name the model explicitly
(`mantle/<model>`), so every condition calls the same model; the router's own
cost is measured separately in the interceptor's logs.

    python3 runner/pipeline_bench.py --condition full --n 60
"""
import argparse
import datetime as dt
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gwclient import Client  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODEL = "mistral.ministral-3-3b-instruct"
PROMPT = "Reply with the single word: ok."


def set_chain(session, param: str, condition: str, ttl: float) -> None:
    ssm = session.client("ssm")
    saved = ROOT / "results" / "pipeline-bench" / "full-config.json"
    saved.parent.mkdir(parents=True, exist_ok=True)
    if not saved.exists():
        saved.write_text(ssm.get_parameter(Name=param)["Parameter"]["Value"])
    full = json.loads(saved.read_text())
    if condition == "empty":
        cfg = {"fail_open": True, "plugins": []}
    elif condition == "no_guardrail":
        cfg = {**full, "plugins": [p for p in full["plugins"] if p["plugin"] != "guardrail"]}
    else:
        cfg = full
    ssm.put_parameter(Name=param, Value=json.dumps(cfg), Overwrite=True, Type="String")
    wait = ttl + 10
    print(f"chain set to {condition!r} ({len(cfg['plugins'])} plugins); waiting {wait:.0f}s for the TTL")
    time.sleep(wait)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", default="gwpipeline")
    ap.add_argument("--condition", required=True, choices=["none", "empty", "no_guardrail", "full"])
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--ttl", type=float, default=30)
    args = ap.parse_args()

    gw = Client.for_stack(args.stack)
    direct = Client("https://bedrock-mantle.us-east-1.api.aws/v1/chat/completions", "bedrock-mantle")
    if args.condition != "none":
        set_chain(gw.session, gw.outputs["PipelineConfigParameter"], args.condition, args.ttl)

    body = lambda model: {"model": model, "max_tokens": 8, "temperature": 0,  # noqa: E731
                          "messages": [{"role": "user", "content": PROMPT}]}
    out = ROOT / "results" / "pipeline-bench" / f"{args.condition}.jsonl"
    rows, started = [], dt.datetime.now(dt.timezone.utc)
    for i in range(args.warmup + args.n):
        ds, _, _, dms = direct.post(body(MODEL))
        gs, _, _, gms = gw.post(body(f"mantle/{MODEL}"), {"x-tenant-id": "bench"})
        if i < args.warmup:
            continue
        rows.append({"condition": args.condition, "i": i - args.warmup, "direct_ms": round(dms, 1),
                     "gateway_ms": round(gms, 1), "direct_status": ds, "gateway_status": gs})
        if (i - args.warmup + 1) % 20 == 0:
            print(f"  {i - args.warmup + 1}/{args.n}")
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out.with_suffix(".window.json")).write_text(json.dumps(
        {"start": started.isoformat(), "end": dt.datetime.now(dt.timezone.utc).isoformat()}))

    diffs = sorted(r["gateway_ms"] - r["direct_ms"] for r in rows if r["gateway_status"] == 200)
    bad = sum(r["gateway_status"] != 200 for r in rows)
    med = diffs[len(diffs) // 2] if diffs else float("nan")
    print(f"{args.condition}: n={len(rows)} non-200={bad}  median overhead {med:.0f} ms  "
          f"(p90 {diffs[int(len(diffs) * 0.9)] if diffs else float('nan'):.0f} ms)")
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
