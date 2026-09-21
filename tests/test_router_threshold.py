#!/usr/bin/env python3
"""The default configuration must route a non-trivial share of traffic to strong.

This exists because it didn't. ROUTER_THRESHOLD defaulted to 0.5 while the
trained scores top out at 0.439, so the `routellm` arm was an `always_weak`
arm with an embedding call attached, and nothing in the system said so.

    python3 tests/test_router_threshold.py
"""
import json
import pathlib
import sys

ART = pathlib.Path(__file__).resolve().parent.parent / "router/artifacts/router_weights.json"


def main() -> int:
    spec = json.loads(ART.read_text())
    ops = spec.get("operating_points")
    fails = []
    if not ops:
        print("FAIL artifact carries no operating_points")
        return 1
    smax, table = ops["score_max_holdout"], ops["thresholds_by_call_rate_pct"]

    if smax >= 0.5:
        fails.append(f"score_max {smax} >= 0.5 — the old default would now work, revisit")
    if "30" not in table:
        fails.append("no threshold for the handler default ROUTER_CALL_RATE_PCT=30")
    for rate, cut in table.items():
        if cut >= smax:
            fails.append(f"threshold {cut} for {rate}% is above max score {smax}")

    for f in fails:
        print("FAIL", f)
    print(f"{'FAILED' if fails else 'OK'} — score_max={smax}, "
          f"{len(table)} operating points, default 30% -> {table.get('30')}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
