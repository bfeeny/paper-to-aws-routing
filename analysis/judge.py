#!/usr/bin/env python3
"""Pairwise LLM-as-judge with position-bias control.

Compares a candidate arm's answers against a baseline arm's answers on the same
prompts. Every pair is judged twice — once in each order — and a verdict counts
only when both orderings agree on the same underlying winner. Disagreement is
recorded as an inconsistency rather than quietly resolved, because the rate of
disagreement is itself a measure of how much to trust the judge.

The judge is called directly (not through the gateway under test) so that
routing cannot influence scoring.

    python3 analysis/judge.py --baseline results/<run-a> --candidate results/<run-b>
"""

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "runner"))
from mantle import Mantle  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent

JUDGE_PROMPT = """You are comparing two answers to the same question.

Question:
{question}

Answer A:
{a}

Answer B:
{b}

Which answer is better? Judge on correctness first, then completeness, then \
clarity. Length is not quality: do not prefer an answer because it is longer. \
Reply with exactly one word: A, B, or TIE."""


def load_run(path: pathlib.Path) -> tuple[dict, dict]:
    manifest = json.loads((path / "manifest.json").read_text())
    rows = {}
    for line in (path / "responses.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("status") == 200 and r.get("response"):
            rows[r["prompt_id"]] = r
    return manifest, rows


def prompts_by_id(prompts_file: str) -> dict:
    rows = [json.loads(l) for l in (ROOT / prompts_file).read_text().splitlines() if l.strip()]
    return {r["id"]: r["prompt"] for r in rows}


def parse_verdict(text: str | None) -> str | None:
    if not text:
        return None
    t = text.strip().upper()
    for token in ("TIE", "A", "B"):
        if t.startswith(token):
            return token
    if "TIE" in t:
        return "TIE"
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="run dir treated as the reference")
    ap.add_argument("--candidate", required=True, help="run dir being scored")
    ap.add_argument("--experiment", default="experiments/pilot.json")
    ap.add_argument("--judge-model", default=None, help="overrides the experiment's judge")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--profile", default="personal")
    ap.add_argument("--region", default="us-east-1")
    args = ap.parse_args()

    cfg = json.loads((ROOT / args.experiment).read_text())
    judge_model = args.judge_model or cfg["judge"]["model"]
    base_dir, cand_dir = ROOT / args.baseline, ROOT / args.candidate
    base_man, base_rows = load_run(base_dir)
    cand_man, cand_rows = load_run(cand_dir)
    questions = prompts_by_id(base_man["prompts_file"])

    shared = sorted(set(base_rows) & set(cand_rows))
    if args.limit:
        shared = shared[: args.limit]
    if not shared:
        print("no overlapping successful prompts between the two runs", file=sys.stderr)
        return 2

    client = Mantle(profile=args.profile, region=args.region)
    out_dir = ROOT / "results" / "judgments" / f"{cand_man['run_id']}__vs__{base_man['run_id']}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"judge     : {judge_model}")
    print(f"baseline  : {base_man['run_id']} ({base_man['arm']})")
    print(f"candidate : {cand_man['run_id']} ({cand_man['arm']})")
    print(f"pairs     : {len(shared)} × 2 orderings\n")

    tally = {"candidate_win": 0, "baseline_win": 0, "tie": 0, "inconsistent": 0, "error": 0}
    records = []
    for i, pid in enumerate(shared, 1):
        q = questions.get(pid, "")
        base_text, cand_text = base_rows[pid]["response"], cand_rows[pid]["response"]

        # order 1: A=baseline, B=candidate   order 2: A=candidate, B=baseline
        s1, t1, _, ms1 = client.complete(
            judge_model, JUDGE_PROMPT.format(question=q, a=base_text, b=cand_text),
            max_tokens=cfg["judge"]["max_tokens"])
        s2, t2, _, ms2 = client.complete(
            judge_model, JUDGE_PROMPT.format(question=q, a=cand_text, b=base_text),
            max_tokens=cfg["judge"]["max_tokens"])
        v1, v2 = parse_verdict(t1), parse_verdict(t2)

        if s1 != 200 or s2 != 200 or v1 is None or v2 is None:
            outcome = "error"
        else:
            # translate each ordering into "who won", independent of position
            w1 = {"A": "baseline", "B": "candidate", "TIE": "tie"}[v1]
            w2 = {"A": "candidate", "B": "baseline", "TIE": "tie"}[v2]
            if w1 == w2:
                outcome = {"baseline": "baseline_win", "candidate": "candidate_win",
                           "tie": "tie"}[w1]
            else:
                outcome = "inconsistent"

        tally[outcome] += 1
        records.append({
            "prompt_id": pid, "outcome": outcome,
            "verdict_order1": v1, "verdict_order2": v2,
            "judge_latency_ms": [round(ms1, 1), round(ms2, 1)],
            "judge_status": [s1, s2],
        })
        print(f"  [{i}/{len(shared)}] {pid:14} {outcome}")

    with (out_dir / "judgments.jsonl").open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")

    decided = tally["candidate_win"] + tally["baseline_win"]
    summary = {
        "baseline_run": base_man["run_id"], "baseline_arm": base_man["arm"],
        "candidate_run": cand_man["run_id"], "candidate_arm": cand_man["arm"],
        "judge_model": judge_model,
        "judge_prompt_sha256": hashlib.sha256(JUDGE_PROMPT.encode()).hexdigest()[:16],
        "dual_ordering": True,
        "pairs": len(shared),
        "tally": tally,
        "win_rate_vs_baseline": (tally["candidate_win"] / decided) if decided else None,
        "inconsistency_rate": tally["inconsistent"] / len(shared),
        "judged_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "note": "win_rate counts only pairs where both orderings agreed on a winner; "
                "ties and inconsistent pairs are excluded from the denominator.",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"\n  {json.dumps(tally)}")
    if decided:
        print(f"  candidate win rate vs baseline: {summary['win_rate_vs_baseline']:.1%} "
              f"({decided} decided pairs)")
    print(f"  inconsistency rate (position bias): {summary['inconsistency_rate']:.1%}")
    print(f"  wrote {out_dir.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
