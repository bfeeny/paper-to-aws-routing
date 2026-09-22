#!/usr/bin/env python3
"""Decompose gateway customization latency into its layers.

Reads the paired runs from runner/pipeline_bench.py and reports, per
condition, the median per-pair overhead (gateway minus direct) with a
percentile-bootstrap interval, plus the interceptor's own per-plugin timings
for the same window from its CloudWatch EMF records.

    python3 analysis/pipeline_overhead.py
"""
import datetime as dt
import json
import pathlib
import random
import statistics
import time

import boto3

ROOT = pathlib.Path(__file__).resolve().parent.parent
RUNS = ROOT / "results" / "pipeline-bench"
ORDER = ["none", "empty", "no_guardrail", "full"]
LOG_GROUP = "/aws/lambda/gwpipeline-interceptor"


def boot_median(xs, n=4000, seed=20260922):
    rng = random.Random(seed)
    meds = sorted(statistics.median(rng.choices(xs, k=len(xs))) for _ in range(n))
    return meds[int(0.025 * n)], meds[int(0.975 * n)]


def plugin_timings(logs, window):
    q = ("filter ispresent(phase) | stats count(*) as calls, "
         "pct(PipelineMs, 50) as pipeline_p50, "
         "pct(tenantMs, 50) as tenant, pct(max_tokensMs, 50) as max_tokens, "
         "pct(routerMs, 50) as router, pct(model_policyMs, 50) as model_policy, "
         "pct(budgetMs, 50) as budget, pct(guardrailMs, 50) as guardrail, "
         "pct(meteringMs, 50) as metering by phase")
    start = int(dt.datetime.fromisoformat(window["start"]).timestamp())
    end = int(dt.datetime.fromisoformat(window["end"]).timestamp()) + 5
    qid = logs.start_query(logGroupName=LOG_GROUP, startTime=start, endTime=end, queryString=q)["queryId"]
    for _ in range(60):
        r = logs.get_query_results(queryId=qid)
        if r["status"] in ("Complete", "Failed", "Cancelled"):
            break
        time.sleep(1)
    return {row_d["phase"]: {k: float(v) for k, v in row_d.items() if k != "phase" and v not in ("",)}
            for row_d in ({c["field"]: c["value"] for c in row} for row in r.get("results", []))}


def main() -> int:
    logs = boto3.Session(profile_name="personal", region_name="us-east-1").client("logs")
    report = {}
    print(f"{'condition':14}{'n':>4}{'median ms':>11}{'95% CI':>18}{'p90':>7}")
    for c in ORDER:
        f = RUNS / f"{c}.jsonl"
        if not f.exists():
            continue
        rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        diffs = [r["gateway_ms"] - r["direct_ms"] for r in rows if r["gateway_status"] == 200]
        lo, hi = boot_median(diffs)
        p90 = sorted(diffs)[int(0.9 * len(diffs))]
        entry = {"n": len(diffs), "median_overhead_ms": round(statistics.median(diffs), 1),
                 "ci95": [round(lo, 1), round(hi, 1)], "p90_ms": round(p90, 1),
                 "direct_median_ms": round(statistics.median(r["direct_ms"] for r in rows), 1)}
        win = f.with_suffix(".window.json")
        if c != "none" and win.exists():
            entry["interceptor"] = plugin_timings(logs, json.loads(win.read_text()))
        report[c] = entry
        print(f"{c:14}{len(diffs):4}{entry['median_overhead_ms']:11.1f}"
              f"{f'[{lo:.0f}, {hi:.0f}]':>18}{p90:7.0f}")
    for c in ("no_guardrail", "full"):
        for phase, t in report.get(c, {}).get("interceptor", {}).items():
            parts = {k: round(v, 1) for k, v in t.items() if k not in ("calls", "pipeline_p50") and v > 0.05}
            print(f"  {c} {phase:8} in-Lambda p50 {t.get('pipeline_p50', 0):6.1f} ms  {parts}")
    out = ROOT / "results" / "reports" / "pipeline-overhead.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
