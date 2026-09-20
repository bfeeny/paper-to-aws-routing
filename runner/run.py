#!/usr/bin/env python3
"""Execute one arm of an experiment and write an auditable run directory.

Output: results/<run-id>/
    manifest.json   what was run — git sha, stack, models, config, timings
    responses.jsonl one record per prompt, including the model that answered
    ledger.json     token totals, quota burndown, and dollars where prices are known

A run is append-only and self-describing: nothing downstream needs to know how it
was produced. Failures are recorded, not dropped, so a partial run is visible as
a partial run.

    python3 runner/run.py --experiment experiments/pilot.json --arm always_strong \
        --stack routingstudyalwaysstrong --split dev --limit 5
"""

import argparse
import datetime as dt
import json
import pathlib
import random
import subprocess
import sys
import time
import urllib.error
import urllib.request

import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

SERVICE = "bedrock-agentcore"
VIRTUAL_MODEL = "auto"
ROOT = pathlib.Path(__file__).resolve().parent.parent


def git_sha() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        return "unknown"


def load_prompts(cfg: dict, split: str) -> list[dict]:
    rows = [json.loads(l) for l in (ROOT / cfg["prompts"]).read_text().splitlines() if l.strip()]
    rng = random.Random(cfg["split"]["seed"])
    order = list(range(len(rows)))
    rng.shuffle(order)
    cut = int(len(rows) * cfg["split"]["dev_fraction"])
    keep = set(order[:cut]) if split == "dev" else set(order[cut:])
    return [r for i, r in enumerate(rows) if i in keep]


def stack_outputs(session, stack: str) -> dict:
    cfn = session.client("cloudformation")
    s = cfn.describe_stacks(StackName=stack)["Stacks"][0]
    return {o["OutputKey"]: o["OutputValue"] for o in s.get("Outputs", [])}


def call(session, url: str, payload: dict, region: str, timeout: int = 180):
    body = json.dumps(payload).encode()
    creds = session.get_credentials().get_frozen_credentials()
    req = AWSRequest(method="POST", url=url, data=body,
                     headers={"Content-Type": "application/json"})
    SigV4Auth(creds, SERVICE, region).add_auth(req)
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(
            urllib.request.Request(url, data=body, headers=dict(req.headers)), timeout=timeout
        ) as r:
            return r.status, json.load(r), (time.perf_counter() - started) * 1000
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read().decode())
        except Exception:  # noqa: BLE001
            detail = {"error": "unparseable"}
        return e.code, detail, (time.perf_counter() - started) * 1000
    except Exception as e:  # noqa: BLE001
        return 0, {"error": repr(e)}, (time.perf_counter() - started) * 1000


def ledger_for(records: list[dict], prices: dict, max_tokens: int) -> dict:
    per_model: dict[str, dict] = {}
    for rec in records:
        if rec["status"] != 200:
            continue
        model = rec["answered_model"] or "unknown"
        usage = rec.get("usage") or {}
        p_in = usage.get("prompt_tokens", 0)
        p_out = usage.get("completion_tokens", 0)
        entry = per_model.setdefault(model, {
            "requests": 0, "input_tokens": 0, "output_tokens": 0,
            "quota_tokens": 0, "usd": 0.0, "priced": True,
        })
        entry["requests"] += 1
        entry["input_tokens"] += p_in
        entry["output_tokens"] += p_out

        spec = prices.get("models", {}).get(model, {})
        weight = spec.get("quota_output_weight") or 1.0
        # Bedrock reserves input + max_tokens up front, then refunds the unused part;
        # the reservation is what competes for quota at peak.
        entry["quota_tokens"] += p_in + int(max_tokens * weight)

        if spec.get("input_per_1k") is None or spec.get("output_per_1k") is None:
            entry["priced"] = False
        else:
            entry["usd"] += (p_in / 1000) * spec["input_per_1k"] + \
                            (p_out / 1000) * spec["output_per_1k"]

    totals = {
        "requests": sum(e["requests"] for e in per_model.values()),
        "input_tokens": sum(e["input_tokens"] for e in per_model.values()),
        "output_tokens": sum(e["output_tokens"] for e in per_model.values()),
        "quota_tokens": sum(e["quota_tokens"] for e in per_model.values()),
        "usd": round(sum(e["usd"] for e in per_model.values()), 6),
        "fully_priced": all(e["priced"] for e in per_model.values()) and bool(per_model),
    }
    for e in per_model.values():
        e["usd"] = round(e["usd"], 6)
    return {"per_model": per_model, "totals": totals,
            "note": "usd is omitted-as-zero where prices.json has nulls; check fully_priced"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--stack", required=True)
    ap.add_argument("--split", choices=["dev", "heldout"], default="dev")
    ap.add_argument("--limit", type=int, default=0, help="0 = all prompts in the split")
    ap.add_argument("--profile", default="personal")
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--label", default="", help="suffix for the run id")
    args = ap.parse_args()

    cfg = json.loads((ROOT / args.experiment).read_text())
    prices = json.loads((ROOT / "experiments/prices.json").read_text())
    prompts = load_prompts(cfg, args.split)
    if args.limit:
        prompts = prompts[: args.limit]

    session = boto3.Session(profile_name=args.profile, region_name=args.region)
    out = stack_outputs(session, args.stack)
    if out.get("RouterStrategy") != args.arm:
        print(f"refusing to run: stack {args.stack} is arm "
              f"{out.get('RouterStrategy')!r}, not {args.arm!r}", file=sys.stderr)
        return 2
    url = out["GatewayUrl"].rstrip("/") + cfg["request"]["path"]

    run_id = (f"{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}-{cfg['name']}"
              f"-{cfg['pair']['name']}-{args.arm}-{args.split}"
              + (f"-{args.label}" if args.label else ""))
    run_dir = ROOT / "results" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"run {run_id}\n  {len(prompts)} prompts -> {url}")
    records = []
    started_at = dt.datetime.now(dt.timezone.utc)
    with (run_dir / "responses.jsonl").open("w") as fh:
        for i, p in enumerate(prompts, 1):
            payload = {
                "model": VIRTUAL_MODEL,
                "messages": [{"role": "user", "content": p["prompt"]}],
                "max_tokens": cfg["request"]["max_tokens"],
                "temperature": cfg["request"]["temperature"],
            }
            status, data, ms = call(session, url, payload, args.region)
            rec = {
                "prompt_id": p["id"],
                "category": p.get("category"),
                "status": status,
                "latency_ms": round(ms, 1),
                "answered_model": data.get("model") if status == 200 else None,
                "usage": data.get("usage") if status == 200 else None,
                "response": (data.get("choices") or [{}])[0].get("message", {}).get("content")
                if status == 200 else None,
                "error": None if status == 200 else json.dumps(data)[:500],
            }
            records.append(rec)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            mark = rec["answered_model"] or f"ERR {status}"
            print(f"  [{i}/{len(prompts)}] {p['id']:14} {ms:7.0f}ms  {mark}")

    finished_at = dt.datetime.now(dt.timezone.utc)
    ok = sum(1 for r in records if r["status"] == 200)
    manifest = {
        "run_id": run_id,
        "experiment": cfg["name"],
        "experiment_file": args.experiment,
        "arm": args.arm,
        "split": args.split,
        "pair": cfg["pair"],
        "stack": args.stack,
        "gateway_url": out["GatewayUrl"],
        "request": cfg["request"],
        "prompts_file": cfg["prompts"],
        "prompt_count": len(prompts),
        "succeeded": ok,
        "failed": len(records) - ok,
        "git_sha": git_sha(),
        "region": args.region,
        "started_utc": started_at.isoformat(timespec="seconds"),
        "finished_utc": finished_at.isoformat(timespec="seconds"),
        "prices_version": prices.get("_priced_on"),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    ledger = ledger_for(records, prices, cfg["request"]["max_tokens"])
    (run_dir / "ledger.json").write_text(json.dumps(ledger, indent=2) + "\n")

    print(f"\n  ok={ok} failed={len(records) - ok}")
    for model, e in ledger["per_model"].items():
        print(f"  {model:42} {e['requests']:4} req  "
              f"in={e['input_tokens']:6} out={e['output_tokens']:6} quota={e['quota_tokens']:7}")
    print(f"  wrote {run_dir.relative_to(ROOT)}")
    return 0 if ok == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
