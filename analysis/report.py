#!/usr/bin/env python3
"""Turn runs + judgments into a report, with uncertainty attached to every number.

A win rate without an interval is a claim without evidence: on 20 prompts, a
60/40 split and a coin flip look identical. Every rate here carries a 95%
bootstrap interval over prompts, and the report states plainly when an interval
spans 50%.

    python3 analysis/report.py --judgment results/judgments/<dir>
"""

import argparse
import json
import pathlib
import random
import statistics

ROOT = pathlib.Path(__file__).resolve().parent.parent


def bootstrap_ci(outcomes: list[str], resamples: int = 10000, seed: int = 20260921):
    """95% CI for candidate win rate among decided pairs."""
    rng = random.Random(seed)
    n = len(outcomes)
    rates = []
    for _ in range(resamples):
        sample = [outcomes[rng.randrange(n)] for _ in range(n)]
        decided = [o for o in sample if o in ("candidate_win", "baseline_win")]
        if decided:
            rates.append(sum(o == "candidate_win" for o in decided) / len(decided))
    if not rates:
        return None, None
    rates.sort()
    return rates[int(0.025 * len(rates))], rates[int(0.975 * len(rates))]


def load(run_dir: pathlib.Path):
    manifest = json.loads((run_dir / "manifest.json").read_text())
    ledger = json.loads((run_dir / "ledger.json").read_text())
    rows = [json.loads(l) for l in (run_dir / "responses.jsonl").read_text().splitlines() if l.strip()]
    return manifest, ledger, rows


def cost_block(manifest, ledger, rows):
    t = ledger["totals"]
    n = max(t["requests"], 1)
    lat = sorted(r["latency_ms"] for r in rows if r["status"] == 200)
    return {
        "arm": manifest["arm"],
        "model": ", ".join(ledger["per_model"].keys()),
        "requests": t["requests"],
        "usd_per_1k_requests": round(t["usd"] / n * 1000, 4) if t["fully_priced"] else None,
        "tokens_per_request": round((t["input_tokens"] + t["output_tokens"]) / n, 1),
        "quota_per_request": round(t["quota_tokens"] / n, 1),
        "quota_inflation": round(t["quota_tokens"] / max(t["input_tokens"] + t["output_tokens"], 1), 2),
        "latency_p50_ms": round(statistics.median(lat)) if lat else None,
        "latency_p90_ms": round(lat[int(0.9 * len(lat))]) if lat else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judgment", required=True, help="results/judgments/<candidate>__vs__<baseline>")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    jdir = ROOT / args.judgment
    summary = json.loads((jdir / "summary.json").read_text())
    judgments = [json.loads(l) for l in (jdir / "judgments.jsonl").read_text().splitlines() if l.strip()]
    outcomes = [j["outcome"] for j in judgments]

    base = ROOT / "results" / summary["baseline_run"]
    cand = ROOT / "results" / summary["candidate_run"]
    b_man, b_led, b_rows = load(base)
    c_man, c_led, c_rows = load(cand)
    b, c = cost_block(b_man, b_led, b_rows), cost_block(c_man, c_led, c_rows)

    n = len(outcomes)
    decided = sum(o in ("candidate_win", "baseline_win") for o in outcomes)
    wins = sum(o == "candidate_win" for o in outcomes)
    ties = sum(o == "tie" for o in outcomes)
    inconsistent = sum(o == "inconsistent" for o in outcomes)
    lo, hi = bootstrap_ci(outcomes)
    win_rate = wins / decided if decided else None

    lines = []
    w = lines.append
    w(f"# {c['arm']} vs {b['arm']} — {b_man['pair']['name']}")
    w("")
    w(f"Prompts: **{n}** ({b_man['split']} split) · judge: `{summary['judge_model']}`, "
      f"both orderings · run {c_man['run_id']}")
    w("")
    w("## Quality")
    w("")
    w("| | value |")
    w("| --- | --- |")
    w(f"| candidate wins | {wins} |")
    w(f"| baseline wins | {decided - wins} |")
    w(f"| ties | {ties} |")
    w(f"| inconsistent (order-dependent) | {inconsistent} |")
    w(f"| **decided pairs** | **{decided} of {n}** |")
    if win_rate is not None:
        w(f"| candidate win rate | **{win_rate:.1%}** (95% CI {lo:.1%}–{hi:.1%}) |")
    w("")
    if lo is not None and lo <= 0.5 <= hi:
        w(f"> The interval spans 50%, so this run does **not** distinguish the two arms. "
          f"With {decided} decided pairs it could not: the sample is too small to resolve "
          f"anything short of a landslide.")
        w("")
    w(f"Position-bias inconsistency: **{inconsistent / n:.0%}** of pairs "
      f"({inconsistent}/{n}) flipped when the answers were swapped.")
    w("")
    w("## Cost and latency")
    w("")
    w("| metric | " + f"{b['arm']} | {c['arm']} |")
    w("| --- | --- | --- |")
    w(f"| model | `{b['model']}` | `{c['model']}` |")
    w(f"| requests | {b['requests']} | {c['requests']} |")
    usd_b = f"${b['usd_per_1k_requests']}" if b["usd_per_1k_requests"] is not None else "unpriced"
    usd_c = f"${c['usd_per_1k_requests']}" if c["usd_per_1k_requests"] is not None else "unpriced"
    w(f"| USD per 1,000 requests | {usd_b} | {usd_c} |")
    w(f"| tokens per request | {b['tokens_per_request']} | {c['tokens_per_request']} |")
    w(f"| quota tokens per request | {b['quota_per_request']} | {c['quota_per_request']} |")
    w(f"| quota inflation vs actual | {b['quota_inflation']}× | {c['quota_inflation']}× |")
    w(f"| latency p50 | {b['latency_p50_ms']} ms | {c['latency_p50_ms']} ms |")
    w(f"| latency p90 | {b['latency_p90_ms']} ms | {c['latency_p90_ms']} ms |")
    w("")
    if b["usd_per_1k_requests"] and c["usd_per_1k_requests"]:
        saving = 1 - c["usd_per_1k_requests"] / b["usd_per_1k_requests"]
        w(f"Routing everything to the weak model would cut spend by **{saving:.0%}** — "
          f"the ceiling on what any router can save on this pair, achieved only by "
          f"giving up whatever quality the strong model adds.")
        w("")
    w("_Dev-split calibration run. Not a held-out result; no router involved._")

    text = "\n".join(lines) + "\n"
    out = ROOT / (args.out or f"results/reports/{jdir.name}.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(text)
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
