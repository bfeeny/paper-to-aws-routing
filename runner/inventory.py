#!/usr/bin/env python3
"""List every AWS resource this study created, and how to delete it.

Nothing here mutates anything. The point is to answer two questions quickly:
"what is this study currently costing me?" and "did a teardown leave anything
behind?" — including resources CloudFormation doesn't own, like the packaging
bucket and log groups that outlive their stacks.

    python3 runner/inventory.py [--profile personal] [--region us-east-1]
"""

import argparse
import json

import boto3
from botocore.exceptions import ClientError

PROJECT = "paper-to-aws-routing"
STACK_PREFIX = "routingstudy"


def stacks(cfn):
    out = []
    paginator = cfn.get_paginator("list_stacks")
    live = ["CREATE_COMPLETE", "UPDATE_COMPLETE", "UPDATE_ROLLBACK_COMPLETE",
            "ROLLBACK_COMPLETE", "CREATE_IN_PROGRESS", "UPDATE_IN_PROGRESS"]
    for page in paginator.paginate(StackStatusFilter=live):
        for s in page["StackSummaries"]:
            if s["StackName"].startswith(STACK_PREFIX):
                out.append((s["StackName"], s["StackStatus"]))
    return sorted(out)


def gateways(session, profile, region):
    """AgentCore gateways, including any orphaned by a failed stack delete.

    Uses the CLI rather than boto3: the bundled botocore may predate the
    bedrock-agentcore-control service model while the CLI already ships it.
    """
    import subprocess
    r = subprocess.run(
        ["aws", "bedrock-agentcore-control", "list-gateways",
         "--profile", profile, "--region", region, "--output", "json"],
        capture_output=True, text=True)
    if r.returncode != 0:
        return [("(could not list gateways)", r.stderr.strip()[:60])]
    items = (json.loads(r.stdout or "{}") or {}).get("items", [])
    return [(g.get("gatewayId") or g.get("name"), g.get("status"))
            for g in items if str(g.get("name", "")).startswith(STACK_PREFIX)]


def buckets(s3):
    found = []
    for b in s3.list_buckets()["Buckets"]:
        name = b["Name"]
        if STACK_PREFIX not in name:
            continue
        try:
            tags = {t["Key"]: t["Value"] for t in
                    s3.get_bucket_tagging(Bucket=name).get("TagSet", [])}
        except ClientError:
            tags = {}
        size = "?"
        found.append((name, tags.get("Project", "-"), size))
    return found


def log_groups(logs):
    out = []
    for prefix in (f"/aws/lambda/{STACK_PREFIX}",):
        p = logs.get_paginator("describe_log_groups")
        for page in p.paginate(logGroupNamePrefix=prefix):
            for g in page["logGroups"]:
                out.append((g["logGroupName"], g.get("storedBytes", 0),
                            g.get("retentionInDays", "never expires")))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="personal")
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    session = boto3.Session(profile_name=args.profile, region_name=args.region)
    cfn, s3, logs = session.client("cloudformation"), session.client("s3"), session.client("logs")

    st, gw, bk, lg = (stacks(cfn), gateways(session, args.profile, args.region),
                      buckets(s3), log_groups(logs))

    if args.json:
        print(json.dumps({"stacks": st, "gateways": gw, "buckets": bk, "log_groups": lg}, indent=2))
        return 0

    print(f"# {PROJECT} — AWS resources in {args.region}\n")
    print(f"CloudFormation stacks ({len(st)})")
    for name, status in st or []:
        print(f"  {name:34} {status}")
    if not st:
        print("  none")

    print(f"\nAgentCore gateways ({len(gw)})")
    for gid, status in gw or []:
        print(f"  {gid:34} {status}")
    if not gw:
        print("  none")

    print(f"\nS3 buckets ({len(bk)})")
    for name, proj, _ in bk or []:
        print(f"  {name:50} Project={proj}")
    if not bk:
        print("  none")

    print(f"\nCloudWatch log groups ({len(lg)})")
    for name, size, ret in lg or []:
        print(f"  {name:46} {size/1024:8.1f} KiB  retention={ret}")
    if not lg:
        print("  none")

    print("\n# Teardown")
    print("  make down-all          # every arm's stack + the packaging bucket")
    for name, _ in st or []:
        print(f"  aws cloudformation delete-stack --stack-name {name}")
    for name, _, _ in bk or []:
        print(f"  aws s3 rb s3://{name} --force")
    for name, _, _ in lg or []:
        print(f"  aws logs delete-log-group --log-group-name {name}")
    print("\n  Note: log groups declared in the stack are deleted with it; any listed")
    print("  after a teardown are orphans worth removing by hand.")
    print("\n  Bedrock model agreements are account-level and are NOT part of this study's")
    print("  resources. They cost nothing when idle; remove one with:")
    print("  aws bedrock delete-foundation-model-agreement --model-id <id>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
