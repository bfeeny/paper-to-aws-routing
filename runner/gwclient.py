"""Minimal SigV4 client for an AgentCore Gateway inference endpoint (and Bedrock direct)."""
import json
import time
import urllib.error
import urllib.request

import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest


def _session(profile: str | None, region: str) -> boto3.Session:
    """The named profile if its credentials resolve, otherwise the default chain.

    The AWS CLI keeps its own SSO token cache, so a profile the CLI can still
    use may be unusable from botocore. Falling back keeps a long benchmark from
    dying on an expired token when working credentials are right there.
    """
    if profile:
        try:
            s = boto3.Session(profile_name=profile, region_name=region)
            s.client("sts").get_caller_identity()
            return s
        except Exception as exc:  # noqa: BLE001
            print(f"profile {profile!r} unusable ({type(exc).__name__}); using default credentials")
    return boto3.Session(region_name=region)


class Client:
    def __init__(self, url: str, service: str, profile: str = "personal", region: str = "us-east-1"):
        self.url, self.service, self.region = url, service, region
        self.session = _session(profile, region)

    @classmethod
    def for_stack(cls, stack: str, **kw):
        s = _session(kw.get("profile", "personal"), kw.get("region", "us-east-1"))
        out = {o["OutputKey"]: o["OutputValue"]
               for o in s.client("cloudformation").describe_stacks(StackName=stack)["Stacks"][0]["Outputs"]}
        c = cls(out["GatewayUrl"].rstrip("/") + "/inference/v1/chat/completions", "bedrock-agentcore", **kw)
        c.outputs = out
        return c

    def post(self, payload: dict, headers: dict | None = None, timeout: int = 120):
        body = json.dumps(payload).encode()
        h = {"Content-Type": "application/json", **(headers or {})}
        req = AWSRequest(method="POST", url=self.url, data=body, headers=h)
        SigV4Auth(self.session.get_credentials().get_frozen_credentials(),
                  self.service, self.region).add_auth(req)
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(urllib.request.Request(self.url, data=body, headers=dict(req.headers)),
                                        timeout=timeout) as r:
                data = r.read()
                return r.status, {k.lower(): v for k, v in r.headers.items()}, data, (time.perf_counter() - t0) * 1000
        except urllib.error.HTTPError as e:
            return e.code, {k.lower(): v for k, v in e.headers.items()}, e.read(), (time.perf_counter() - t0) * 1000
