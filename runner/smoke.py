#!/usr/bin/env python3
"""Send one request through the gateway and report what happened.

Proves the whole path: SigV4 auth -> gateway -> REQUEST interceptor rewrites the
virtual model -> bedrock-mantle -> response. Prints the model that actually
answered, so a routing decision is visible rather than assumed.

    python3 runner/smoke.py --stack routingstudyalwaysstrong
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

VIRTUAL_MODEL = "auto"
SERVICE = "bedrock-agentcore"


def stack_outputs(session, stack: str) -> dict:
    cfn = session.client("cloudformation")
    stacks = cfn.describe_stacks(StackName=stack)["Stacks"]
    return {o["OutputKey"]: o["OutputValue"] for o in stacks[0].get("Outputs", [])}


def call(session, url: str, payload: dict, region: str) -> tuple[int, dict, float]:
    body = json.dumps(payload).encode()
    creds = session.get_credentials().get_frozen_credentials()
    req = AWSRequest(method="POST", url=url, data=body,
                     headers={"Content-Type": "application/json"})
    SigV4Auth(creds, SERVICE, region).add_auth(req)

    started = time.perf_counter()
    try:
        with urllib.request.urlopen(
            urllib.request.Request(url, data=body, headers=dict(req.headers)), timeout=120
        ) as resp:
            elapsed = (time.perf_counter() - started) * 1000
            return resp.status, json.load(resp), elapsed
    except urllib.error.HTTPError as e:
        elapsed = (time.perf_counter() - started) * 1000
        detail = e.read().decode()[:600]
        return e.code, {"error": detail}, elapsed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", required=True)
    ap.add_argument("--profile", default="personal")
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--prompt", default="In one sentence: what is a pseudo-terminal?")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--model", default=VIRTUAL_MODEL,
                    help="Override to call a concrete model and bypass the router.")
    args = ap.parse_args()

    session = boto3.Session(profile_name=args.profile, region_name=args.region)
    out = stack_outputs(session, args.stack)
    url = out["GatewayUrl"].rstrip("/") + "/inference/v1/chat/completions"

    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": args.max_tokens,
    }

    print(f"arm       : {out.get('RouterStrategy', '?')}")
    print(f"endpoint  : {url}")
    print(f"sent model: {args.model}")

    status, data, elapsed = call(session, url, payload, args.region)
    print(f"status    : {status}  ({elapsed:.0f} ms end-to-end)")

    if status != 200:
        print("error     :", data.get("error"))
        return 1

    usage = data.get("usage", {})
    choice = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
    print(f"answered  : {data.get('model')}")      # the routing decision, observed
    print(f"usage     : {json.dumps(usage)}")
    print(f"reply     : {choice.strip()[:200]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
