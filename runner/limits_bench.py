#!/usr/bin/env python3
"""Show that counting requests and counting tokens are different limits.

Identical traffic — ten requests, each reserving 4,000 output tokens — is run
against two configurations of the same plugin:

  requests   a limit of 10 requests per minute. Every request fits, and the
             tenant has committed 40,000 tokens of the model's quota.
  tokens     a limit of 5,000 tokens per minute. The same traffic is cut off
             almost immediately, because the reservation is what counts.

A third arm sends small requests under the token limit to show the reservation
being reconciled: ask for 4,000, use 30, and the other 3,970 come back.

    python3 runner/limits_bench.py --arm tokens
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

ARMS = {
    # A generous request limit and no token limit: the shape most gateways ship.
    "requests": {"requests_per_min": {"*": 10}},
    # The same traffic, limited by what the quota is actually denominated in.
    "tokens": {"tokens_per_min": {"*": 5000}},
    # Both, plus the sliding-window approximation.
    "smooth": {"requests_per_min": {"*": 10}, "tokens_per_min": {"*": 5000}, "smooth": True},
}


def configure(session, param: str, arm: str, ttl: float) -> str:
    ssm = session.client("ssm")
    before = ssm.get_parameter(Name=param)["Parameter"]["Value"]
    cfg = json.loads(before)
    plugins = [p for p in cfg["plugins"] if p["plugin"] not in ("rate_limit", "budget", "guardrail")]
    at = next((i for i, p in enumerate(plugins) if p["plugin"] == "tenant"), -1) + 1
    plugins.insert(at, {"plugin": "rate_limit", "params": ARMS[arm]})
    cfg["plugins"] = plugins
    ssm.put_parameter(Name=param, Value=json.dumps(cfg), Overwrite=True, Type="String")
    print(f"arm={arm} {ARMS[arm]}; budget and guardrail out of the chain; "
          f"waiting {ttl + 10:.0f}s for the config TTL")
    time.sleep(ttl + 10)
    return before


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", default="gwpipeline")
    ap.add_argument("--arm", default="tokens", choices=sorted(ARMS))
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--ttl", type=float, default=30)
    ap.add_argument("--concurrency", type=int, default=1,
                    help="send this many at once; reservations then overlap")
    args = ap.parse_args()

    gw = Client.for_stack(args.stack)
    restore = configure(gw.session, gw.outputs["PipelineConfigParameter"], args.arm, args.ttl)
    # A fresh tenant per run starts both windows empty.
    tenant = f"rl-{args.arm}-{dt.datetime.now(dt.timezone.utc):%H%M%S}"
    rows, reserved_total = [], 0

    def one(i: int) -> dict:
        # A unique prompt per request: an exact-cache hit would be served from
        # the interceptor and release its reservation, which is correct
        # behavior and would hide what this bench is measuring.
        status, _, raw, ms = gw.post(
            {"model": f"mantle/{MODEL}", "max_tokens": args.max_tokens, "temperature": 0,
             "messages": [{"role": "user", "content": f"Reply with the single word: ok. #{i}"}]},
            {"x-tenant-id": tenant})
        body = json.loads(raw) if raw else {}
        usage = body.get("usage") or {}
        return {"i": i, "status": status, "code": (body.get("error") or {}).get("code"),
                "ms": round(ms, 1),
                "used": usage.get("total_tokens")
                or (usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0))}

    try:
        if args.concurrency > 1:
            import concurrent.futures as cf
            with cf.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                rows = sorted(pool.map(one, range(args.n)), key=lambda r: r["i"])
        else:
            rows = [one(i) for i in range(args.n)]
        reserved_total = sum(args.max_tokens for r in rows if r["status"] == 200)
        for r in rows:
            print(f"  {r['i']:2} {r['status']} {r['code'] or 'ok':22} {r['ms']:7.0f} ms  used={r['used']}")
    finally:
        gw.session.client("ssm").put_parameter(
            Name=gw.outputs["PipelineConfigParameter"], Value=restore, Overwrite=True, Type="String")

    allowed = [r for r in rows if r["status"] == 200]
    used = sum(r["used"] or 0 for r in allowed)
    summary = {
        "arm": args.arm, "limits": ARMS[args.arm], "tenant": tenant,
        "concurrency": args.concurrency, "max_tokens_each": args.max_tokens,
        "requests_sent": len(rows), "requests_allowed": len(allowed),
        "tokens_reserved_by_allowed": reserved_total,
        "tokens_actually_used": used,
        "first_refusal_at": next((r["i"] for r in rows if r["status"] != 200), None),
        "refusal_code": next((r["code"] for r in rows if r["status"] != 200), None),
        "window": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    out = (ROOT / "results" / "reports"
           / f"rate-limit-{args.arm}-c{args.concurrency}.json")
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1))
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
