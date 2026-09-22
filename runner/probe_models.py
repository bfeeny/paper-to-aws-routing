#!/usr/bin/env python3
"""Map which models the mantle endpoint *lists* vs which it will actually serve.

`/v1/models` advertises the catalog; an account may be entitled to only part of
it, and the API surface differs by family (OpenAI-compatible chat/completions for
most, the Anthropic messages API for Claude). This probe records, per model, the
status of a minimal call on each supported path.

Writes JSONL to results/model-availability/<date>.jsonl.

    python3 runner/probe_models.py --profile personal
"""

import argparse
import concurrent.futures as cf
import datetime as dt
import json
import pathlib
import time
import urllib.error
import urllib.request

import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

SERVICE = "bedrock-mantle"
PATHS = {
    "chat_completions": "/v1/chat/completions",
    "responses": "/v1/responses",
    "anthropic_messages": "/anthropic/v1/messages",
}


def signed_post(creds, region, url, payload, timeout=60):
    body = json.dumps(payload).encode()
    req = AWSRequest(method="POST", url=url, data=body,
                     headers={"Content-Type": "application/json"})
    SigV4Auth(creds, SERVICE, region).add_auth(req)
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(
            urllib.request.Request(url, data=body, headers=dict(req.headers)), timeout=timeout
        ) as r:
            data = json.load(r)
            return r.status, data, (time.perf_counter() - started) * 1000
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read().decode())
        except Exception:  # noqa: BLE001
            detail = {"raw": "unparseable"}
        return e.code, detail, (time.perf_counter() - started) * 1000
    except Exception as e:  # noqa: BLE001 - network/timeouts are data too
        return 0, {"error": repr(e)}, (time.perf_counter() - started) * 1000


def payload_for(kind: str, model: str) -> dict:
    msgs = [{"role": "user", "content": "Say OK."}]
    if kind == "responses":
        return {"model": model, "input": "Say OK.", "max_output_tokens": 1}
    return {"model": model, "messages": msgs, "max_tokens": 1}


def error_reason(detail: dict) -> str:
    err = detail.get("error", detail)
    if isinstance(err, dict):
        return (err.get("message") or err.get("code") or "")[:200]
    return str(err)[:200]


def probe(creds, region, base, model):
    row = {"model": model, "paths": {}}
    for kind, path in PATHS.items():
        status, detail, ms = signed_post(creds, region, base + path, payload_for(kind, model))
        entry = {"status": status, "latency_ms": round(ms, 1)}
        if status == 200:
            entry["usage"] = detail.get("usage")
        else:
            entry["reason"] = error_reason(detail)
        row["paths"][kind] = entry
    row["callable"] = [k for k, v in row["paths"].items() if v["status"] == 200]
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="personal")
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    base = f"https://bedrock-mantle.{args.region}.api.aws"
    session = boto3.Session(profile_name=args.profile, region_name=args.region)
    creds = session.get_credentials().get_frozen_credentials()

    # Catalog
    listing_url = base + "/v1/models"
    req = AWSRequest(method="GET", url=listing_url, headers={})
    SigV4Auth(creds, SERVICE, args.region).add_auth(req)
    with urllib.request.urlopen(
        urllib.request.Request(listing_url, headers=dict(req.headers)), timeout=30
    ) as r:
        listed = sorted(m["id"] for m in json.load(r)["data"])

    print(f"listed: {len(listed)} models; probing {len(PATHS)} paths each")
    rows = []
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        for row in ex.map(lambda m: probe(creds, args.region, base, m), listed):
            rows.append(row)
            state = ",".join(row["callable"]) or "NONE"
            print(f"  {row['model']:42} {state}")

    out = pathlib.Path(args.out or
                       f"results/model-availability/{dt.date.today().isoformat()}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    callable_models = [r["model"] for r in rows if r["callable"]]
    print(f"\nlisted={len(listed)} callable={len(callable_models)} "
          f"unavailable={len(listed) - len(callable_models)}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
