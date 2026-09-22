#!/usr/bin/env python3
"""Report what this study actually cost, from AWS's own billing data.

Two views, because they answer different questions:

  * **Billed** — Cost Explorer, grouped by usage type. Authoritative, and the only
    source that knows the prices of models the pricing APIs don't publish (the
    current Claude models are not in the Price List API or the Marketplace offer
    API). Lags by up to ~24h.
  * **Metered** — token counts from our own run directories and label files.
    Immediate, and useful for projecting a bigger run before paying for it.

    python3 runner/cost_report.py --days 2
"""

import argparse
import collections
import datetime as dt
import json
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent.parent


def aws_json(*args, profile: str, region: str = "us-east-1"):
    r = subprocess.run(["aws", *args, "--profile", profile, "--region", region,
                        "--output", "json"], capture_output=True, text=True)
    if r.returncode != 0:
        return None, r.stderr.strip()[:200]
    return json.loads(r.stdout or "{}"), None


def billed(profile: str, days: int):
    end = dt.date.today() + dt.timedelta(days=1)
    start = end - dt.timedelta(days=days + 1)
    d, err = aws_json(
        "ce", "get-cost-and-usage",
        "--time-period", f"Start={start.isoformat()},End={end.isoformat()}",
        "--granularity", "DAILY", "--metrics", "UnblendedCost", "UsageQuantity",
        "--filter", json.dumps({"Dimensions": {"Key": "SERVICE",
                                               "Values": ["Amazon Bedrock"]}}),
        "--group-by", "Type=DIMENSION,Key=USAGE_TYPE",
        profile=profile)
    if err:
        return None, err
    per_day, per_usage = {}, collections.defaultdict(float)
    for period in d.get("ResultsByTime", []):
        day = period["TimePeriod"]["Start"]
        total = 0.0
        for g in period.get("Groups", []):
            amt = float(g["Metrics"]["UnblendedCost"]["Amount"])
            total += amt
            per_usage[g["Keys"][0]] += amt
        per_day[day] = total
    return {"per_day": per_day, "per_usage": dict(per_usage)}, None


def metered():
    """Token counts from everything this study has produced locally."""
    out = collections.defaultdict(lambda: {"calls": 0, "in": 0, "out": 0})

    for ledger in (ROOT / "results").glob("*/ledger.json"):
        data = json.loads(ledger.read_text())
        for model, e in data.get("per_model", {}).items():
            out[model]["calls"] += e.get("requests", 0)
            out[model]["in"] += e.get("input_tokens", 0)
            out[model]["out"] += e.get("output_tokens", 0)

    cascade = ROOT / "experiments" / "cascade_labels.jsonl"
    if cascade.exists():
        for line in cascade.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            for tier, t in rec.get("tiers", {}).items():
                out[f"claude-{tier} (labeling)"]["calls"] += 1
                out[f"claude-{tier} (labeling)"]["in"] += t.get("in") or 0
                out[f"claude-{tier} (labeling)"]["out"] += t.get("out") or 0
    return dict(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=2)
    ap.add_argument("--profile", default="personal")
    args = ap.parse_args()

    print("# Metered locally (immediate; from run ledgers and label files)\n")
    m = metered()
    tot_in = tot_out = tot_calls = 0
    for model in sorted(m, key=lambda k: -(m[k]["in"] + m[k]["out"])):
        e = m[model]
        tot_calls += e["calls"]; tot_in += e["in"]; tot_out += e["out"]
        print(f"  {model:46} {e['calls']:6} calls  in={e['in']:8}  out={e['out']:8}")
    print(f"  {'TOTAL':46} {tot_calls:6} calls  in={tot_in:8}  out={tot_out:8}")

    print("\n# Billed by AWS (authoritative; Cost Explorer lags up to ~24h)\n")
    b, err = billed(args.profile, args.days)
    if err:
        print(f"  unavailable: {err}")
        return 0
    for day, amt in sorted(b["per_day"].items()):
        print(f"  {day}  ${amt:,.4f}")
    print()
    for usage, amt in sorted(b["per_usage"].items(), key=lambda kv: -kv[1])[:16]:
        if amt > 0:
            print(f"  {usage:58} ${amt:,.4f}")
    print(f"\n  Bedrock total over {args.days} day(s): "
          f"${sum(b['per_day'].values()):,.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
