#!/usr/bin/env python3
"""Compare cache backends behind the same interceptor.

Switches the cache plugin's backend through the SSM chain rather than
redeploying, so the Lambda, its VPC placement and every other plugin stay
identical and only the store changes. Reports client-side hit latency and the
plugin's own in-Lambda time from CloudWatch.

    python3 runner/cache_bench.py --n 30
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


def set_backend(session, param: str, backend: str, ttl: float):
    ssm = session.client("ssm")
    cfg = json.loads(ssm.get_parameter(Name=param)["Parameter"]["Value"])
    for p in cfg["plugins"]:
        if p["plugin"] == "cache":
            p.setdefault("params", {})["backend"] = backend
    ssm.put_parameter(Name=param, Value=json.dumps(cfg), Overwrite=True, Type="String")
    time.sleep(ttl + 10)


def plugin_ms(logs, start, end, group):
    q = ("filter phase = 'request' and ispresent(cacheMs) "
         "| stats pct(cacheMs, 50) as cache_p50, count(*) as n")
    qid = logs.start_query(logGroupName=group, startTime=int(start), endTime=int(end) + 5,
                           queryString=q)["queryId"]
    for _ in range(40):
        r = logs.get_query_results(queryId=qid)
        if r["status"] in ("Complete", "Failed", "Cancelled"):
            break
        time.sleep(1)
    rows = [{c["field"]: c["value"] for c in row} for row in r.get("results", [])]
    return float(rows[0]["cache_p50"]) if rows and rows[0].get("cache_p50") else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", default="gwpipeline")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--ttl", type=float, default=30)
    args = ap.parse_args()

    gw = Client.for_stack(args.stack)
    logs = gw.session.client("logs")
    group = f"/aws/lambda/{gw.outputs['InterceptorFunction']}"
    out = {}
    for backend in ("dynamodb", "valkey"):
        set_backend(gw.session, gw.outputs["PipelineConfigParameter"], backend, args.ttl)
        body = lambda q: {"model": "auto", "max_tokens": 32, "temperature": 0,  # noqa: E731
                          "messages": [{"role": "user", "content": q}]}
        q = f"Name one ocean. Three words. ({backend} {int(time.time())})"
        gw.post(body(q), {"x-tenant-id": "cachebench"}, timeout=90)   # populate
        for _ in range(3):
            gw.post(body(q), {"x-tenant-id": "cachebench"})           # warm the container
        start, hits = dt.datetime.now(dt.timezone.utc).timestamp(), []
        for _ in range(args.n):
            st, _, raw, ms = gw.post(body(q), {"x-tenant-id": "cachebench"})
            if st == 200 and json.loads(raw).get("x_gateway", {}).get("cached"):
                hits.append(ms)
        end = dt.datetime.now(dt.timezone.utc).timestamp()
        time.sleep(20)   # let the logs land
        out[backend] = {"hits": len(hits), "median_ms": round(statistics.median(hits), 1),
                        "p10_ms": round(sorted(hits)[len(hits) // 10], 1),
                        "in_lambda_p50_ms": round(plugin_ms(logs, start, end, group), 2)}
        print(f"{backend:9} hits={out[backend]['hits']:3} client median "
              f"{out[backend]['median_ms']:6.0f} ms   plugin p50 {out[backend]['in_lambda_p50_ms']:5.2f} ms")
    path = ROOT / "results" / "reports" / "cache-backends.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
