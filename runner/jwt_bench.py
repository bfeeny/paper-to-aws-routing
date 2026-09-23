#!/usr/bin/env python3
"""Call the token-authorized gateway as two different tenants.

Fetches a client-credentials token per tenant from Cognito and sends it as a
bearer token. Nothing in the request names the tenant: it is a claim in a token
the gateway validated before the interceptor ever ran.

The point of the paired arms is what a header can and cannot do any more:

    python3 runner/jwt_bench.py --check          # both tenants, and the spoof attempt
    python3 runner/jwt_bench.py --verify false   # what skipping local verification saves
"""
import argparse
import base64
import datetime as dt
import json
import pathlib
import statistics
import sys
import time
import urllib.parse
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gwclient import Client, _session  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODEL = "mistral.ministral-3-3b-instruct"


def outputs(stack: str, profile: str = "personal") -> dict:
    s = _session(profile, "us-east-1")
    d = s.client("cloudformation").describe_stacks(StackName=stack)["Stacks"][0]["Outputs"]
    return {o["OutputKey"]: o["OutputValue"] for o in d}


def client_secret(session, pool_id: str, client_id: str) -> str:
    idp = session.client("cognito-idp")
    return idp.describe_user_pool_client(
        UserPoolId=pool_id, ClientId=client_id)["UserPoolClient"]["ClientSecret"]


def token_for(token_url: str, client_id: str, secret: str, scope: str) -> tuple[str, float]:
    """Client-credentials grant: no user, no redirect, just a machine identity."""
    data = urllib.parse.urlencode({"grant_type": "client_credentials", "scope": scope}).encode()
    basic = base64.b64encode(f"{client_id}:{secret}".encode()).decode()
    req = urllib.request.Request(token_url, data=data, headers={
        "Content-Type": "application/x-www-form-urlencoded",
        "Authorization": f"Basic {basic}"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=10) as r:
        body = json.load(r)
    return body["access_token"], (time.perf_counter() - t0) * 1000


def claims_of(token: str) -> dict:
    payload = token.split(".")[1]
    payload += "=" * (-len(payload) % 4)
    return json.loads(base64.urlsafe_b64decode(payload))


def ask(url: str, token: str | None, text: str, extra: dict | None = None):
    body = json.dumps({"model": f"mantle/{MODEL}", "max_tokens": 32, "temperature": 0,
                       "messages": [{"role": "user", "content": text}]}).encode()
    headers = {"Content-Type": "application/json", **(extra or {})}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=body, headers=headers)
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            raw, status = r.read(), r.status
    except urllib.error.HTTPError as e:
        raw, status = e.read(), e.code
    ms = (time.perf_counter() - t0) * 1000
    try:
        parsed = json.loads(raw)
    except Exception:  # noqa: BLE001
        parsed = {}
    return {"status": status, "ms": round(ms, 1),
            "x_gateway": parsed.get("x_gateway"),
            "error": (parsed.get("error") or {}).get("code") if status != 200 else None}


def set_plugin(session, param: str, verify: bool, ttl: float) -> str:
    ssm = session.client("ssm")
    before = ssm.get_parameter(Name=param)["Parameter"]["Value"]
    cfg = json.loads(before)
    plugins = [p for p in cfg["plugins"] if p["plugin"] not in ("tenant", "jwt_tenant")]
    plugins.insert(0, {"plugin": "jwt_tenant", "params": {
        "verify": verify, "claim": "client_id", "scope_prefix": "tenant-",
        "known": ["acme", "globex"]}})
    cfg["plugins"] = plugins
    ssm.put_parameter(Name=param, Value=json.dumps(cfg), Overwrite=True, Type="String")
    print(f"jwt_tenant verify={verify}; waiting {ttl + 10:.0f}s for the config TTL")
    time.sleep(ttl + 10)
    return before


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", default="gwpipeline")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--verify", default="true", choices=["true", "false"])
    ap.add_argument("--ttl", type=float, default=30)
    ap.add_argument("--check", action="store_true", help="identity checks only, no timing run")
    args = ap.parse_args()
    verify = args.verify == "true"

    out = outputs(args.stack)
    session = _session("personal", "us-east-1")
    pool_id = out["JwtIssuer"].rsplit("/", 1)[-1]
    url = out["JwtGatewayUrl"].rstrip("/") + "/inference/v1/chat/completions"

    tenants = {}
    for name, key in (("acme", "TenantAcmeClientId"), ("globex", "TenantGlobexClientId")):
        cid = out[key]
        secret = client_secret(session, pool_id, cid)
        token, token_ms = token_for(out["TokenUrl"], cid, secret, f"gw/tenant-{name}")
        tenants[name] = {"client_id": cid, "token": token, "token_ms": round(token_ms, 1),
                         "claims": {k: v for k, v in claims_of(token).items()
                                    if k in ("client_id", "scope", "token_use", "exp")}}
        print(f"{name}: client {cid}, token in {token_ms:.0f} ms, claims {tenants[name]['claims']}")

    restore = set_plugin(session, out["PipelineConfigParameter"], verify, args.ttl)
    results = {"verify": verify, "tenants": {k: v["claims"] for k, v in tenants.items()}}
    try:
        checks = {}
        checks["acme_token"] = ask(url, tenants["acme"]["token"], "Say ok.")
        checks["globex_token"] = ask(url, tenants["globex"]["token"], "Say ok.")
        # The header the pipeline used to trust, now contradicted by the token.
        checks["acme_token_claiming_globex"] = ask(
            url, tenants["acme"]["token"], "Say ok.", {"x-tenant-id": "globex"})
        checks["no_token"] = ask(url, None, "Say ok.")
        checks["garbage_token"] = ask(url, "not.a.token", "Say ok.")
        # A token that is valid, but minted for a different audience/issuer.
        checks["expired_shape"] = ask(url, tenants["acme"]["token"][:-4] + "AAAA", "Say ok.")
        results["checks"] = checks
        for name, r in checks.items():
            who = (r["x_gateway"] or {}).get("tenant")
            print(f"  {name:28} {r['status']}  tenant={who}  {r['error'] or ''}")

        if not args.check:
            lat = []
            for _ in range(args.n):
                r = ask(url, tenants["acme"]["token"], "Say ok.")
                if r["status"] == 200:
                    lat.append(r["ms"])
            results["median_ms"] = round(statistics.median(lat), 1) if lat else None
            results["n"] = len(lat)
            print(f"  median end-to-end with verify={verify}: {results['median_ms']} ms (n={len(lat)})")
    finally:
        session.client("ssm").put_parameter(
            Name=out["PipelineConfigParameter"], Value=restore, Overwrite=True, Type="String")

    results["window"] = dt.datetime.now(dt.timezone.utc).isoformat()
    path = ROOT / "results" / "reports" / f"jwt-tenancy-{'verify' if verify else 'noverify'}.json"
    path.write_text(json.dumps(results, indent=1))
    print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
