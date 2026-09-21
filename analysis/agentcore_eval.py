#!/usr/bin/env python3
"""Score a run with Amazon Bedrock AgentCore Evaluations (pointwise, managed).

This is the study's *second opinion*. The primary metric is the pairwise judge in
`judge.py`, whose prompt is versioned in this repo; AgentCore's built-in evaluators
are a managed service whose model and rubric can change, which makes them excellent
corroboration and poor provenance. Where the two methods agree, a result is robust;
where they disagree, that disagreement is itself worth reporting.

Practical notes discovered the hard way:
  * `evaluate` only accepts spans whose `scope.name` is a supported instrumentation
    library (Strands, LangChain, OpenInference, LlamaIndex, …). Arbitrary OTel spans
    are rejected. We therefore emit Strands-shaped spans.
  * Message content belongs in span *events* (`gen_ai.user.message`, `gen_ai.choice`),
    not attributes, and should be plain text — JSON-wrapped content confuses the judge.
  * TRACE-level evaluators (Correctness, Helpfulness) target `traceIds`; only
    TOOL_CALL-level evaluators accept `spanIds`.
  * At most 10 evaluations per call.

    python3 analysis/agentcore_eval.py --run results/<run-id>
"""

import argparse
import datetime as dt
import json
import pathlib
import statistics
import subprocess
import time
import uuid

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCOPE = {"name": "strands.telemetry.tracer", "version": "1.0.0"}
BATCH = 10


def spans_for(prompt: str, answer: str, model: str, session_id: str, trace_id: str):
    """A minimal Strands-shaped agent+model trace for one request/response."""
    now = int(time.time() * 1_000_000_000)
    started = now - 1_000_000_000
    events = [
        {"name": "gen_ai.user.message", "timestamp": started, "attributes": {"content": prompt}},
        {"name": "gen_ai.choice", "timestamp": now,
         "attributes": {"message": answer, "finish_reason": "end_turn"}},
    ]

    def span(name, span_id, parent, extra):
        return {
            "name": name,
            "context": {"trace_id": trace_id, "span_id": span_id},
            "parent_id": parent,
            "start_time": started, "end_time": now,
            "status": {"status_code": "OK"}, "kind": "SpanKind.INTERNAL",
            "attributes": {"session.id": session_id, "gen_ai.system": "strands-agents", **extra},
            "events": events,
            "scope": SCOPE,
            "resource": {"attributes": {"service.name": "paper-to-aws-routing"}},
        }

    agent_id, model_id = uuid.uuid4().hex[:16], uuid.uuid4().hex[:16]
    return [
        span("invoke_agent routing-study", agent_id, None,
             {"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": "routing-study",
              "gen_ai.request.model": model}),
        span(f"chat {model}", model_id, agent_id,
             {"gen_ai.operation.name": "chat", "gen_ai.request.model": model}),
    ]


def evaluate(evaluator: str, session_spans: list, trace_ids: list[str],
             profile: str, region: str):
    r = subprocess.run(
        ["aws", "bedrock-agentcore", "evaluate",
         "--evaluator-id", evaluator,
         "--evaluation-input", json.dumps({"sessionSpans": session_spans}),
         "--evaluation-target", json.dumps({"traceIds": trace_ids}),
         "--profile", profile, "--region", region, "--output", "json"],
        capture_output=True, text=True)
    if r.returncode != 0:
        return None, r.stderr.strip()[:300]
    return json.loads(r.stdout or "{}"), None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run directory to score")
    ap.add_argument("--evaluators", default="Builtin.Correctness,Builtin.Helpfulness")
    ap.add_argument("--experiment", default="experiments/pilot.json")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--profile", default="personal")
    ap.add_argument("--region", default="us-east-1")
    args = ap.parse_args()

    run_dir = ROOT / args.run
    manifest = json.loads((run_dir / "manifest.json").read_text())
    cfg = json.loads((ROOT / args.experiment).read_text())
    questions = {
        json.loads(l)["id"]: json.loads(l)["prompt"]
        for l in (ROOT / cfg["prompts"]).read_text().splitlines() if l.strip()
    }

    rows = [json.loads(l) for l in (run_dir / "responses.jsonl").read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("status") == 200 and r.get("response")]
    if args.limit:
        rows = rows[: args.limit]

    out_dir = ROOT / "results" / "evaluations" / manifest["run_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"run       : {manifest['run_id']} ({manifest['arm']})")
    print(f"responses : {len(rows)}")

    summary = {}
    for evaluator in [e.strip() for e in args.evaluators.split(",") if e.strip()]:
        print(f"\n{evaluator}")
        scored, failures, tokens = [], [], 0
        for start in range(0, len(rows), BATCH):
            chunk = rows[start:start + BATCH]
            session_id = f"{manifest['run_id']}-{start // BATCH}"
            spans, trace_ids, by_trace = [], [], {}
            for r in chunk:
                tid = uuid.uuid4().hex
                spans += spans_for(questions.get(r["prompt_id"], ""), r["response"],
                                   r["answered_model"] or "unknown", session_id, tid)
                trace_ids.append(tid)
                by_trace[tid] = r["prompt_id"]

            data, err = evaluate(evaluator, spans, trace_ids, args.profile, args.region)
            if err:
                failures.append(err)
                print(f"  batch {start // BATCH}: FAILED {err[:100]}")
                continue
            for res in data.get("evaluationResults", []):
                tid = res.get("context", {}).get("spanContext", {}).get("traceId")
                usage = res.get("tokenUsage") or {}
                tokens += usage.get("totalTokens", 0)
                rec = {
                    "prompt_id": by_trace.get(tid),
                    "evaluator": evaluator,
                    "value": res.get("value"),
                    "label": res.get("label"),
                    "explanation": res.get("explanation"),
                    "token_usage": usage,
                }
                scored.append(rec)
                print(f"  {rec['prompt_id']:14} {rec['value']}  {rec['label']}")

        with (out_dir / f"{evaluator.replace('.', '_')}.jsonl").open("w") as fh:
            for rec in scored:
                fh.write(json.dumps(rec) + "\n")

        values = [s["value"] for s in scored if isinstance(s.get("value"), (int, float))]
        summary[evaluator] = {
            "scored": len(scored),
            "failed_batches": len(failures),
            "mean": round(statistics.fmean(values), 4) if values else None,
            "stdev": round(statistics.stdev(values), 4) if len(values) > 1 else None,
            "judge_tokens": tokens,
        }

    meta = {
        "run_id": manifest["run_id"], "arm": manifest["arm"], "pair": manifest["pair"],
        "evaluated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "evaluators": summary,
        "note": "Managed evaluators: model and rubric are controlled by AWS and may change. "
                "Treat as corroboration for the versioned pairwise judge, not as provenance.",
    }
    (out_dir / "summary.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"\n{json.dumps(summary, indent=2)}")
    print(f"wrote {out_dir.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
