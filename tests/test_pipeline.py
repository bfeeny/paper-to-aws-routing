#!/usr/bin/env python3
"""Offline tests for the gateway plugin pipeline. No AWS calls: clients are faked.

    python3 tests/test_pipeline.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from gateway import pipeline as P  # noqa: E402
from gateway import plugins  # noqa: E402,F401  (registers plugins)
from gateway.plugins import guardrail as G  # noqa: E402
from gateway import state as S  # noqa: E402

WEAK, STRONG = "mistral.ministral-3-3b-instruct", "mistral.mistral-large-3-675b-instruct"


class FakeStore:
    """In-memory stand-in for state.DynamoStore, same interface and idempotency."""
    def __init__(self): self.spend, self.reqs, self.settled, self.cache = {}, {}, set(), {}
    def get_cached(self, key): return self.cache.get(key)
    def put_cached(self, key, body, ttl_s=0): self.cache[key] = body
    def spend_today(self, tenant): return self.spend.get(tenant, 0.0)
    def remember(self, rid, data): self.reqs[rid] = dict(data)
    def recall(self, rid): return self.reqs.get(rid, {})
    def settle(self, rid, tenant, cost):
        if rid in self.settled:
            return False
        self.settled.add(rid)
        self.spend[tenant] = self.spend.get(tenant, 0.0) + cost
        return True


class FakeBedrock:
    def apply_guardrail(self, content, **_):
        text = content[0]["text"]["text"]
        if "ssn" in text.lower():
            return {"action": "GUARDRAIL_INTERVENED", "outputs": [{"text": "Sorry, that request was blocked."}]}
        return {"action": "NONE"}


def pipe(*specs):
    return P.Pipeline.from_config({"plugins": [{"plugin": n, "params": p} for n, p in specs]})


CHAIN = [
    ("tenant", {"known": ["acme", "globex"]}),
    ("max_tokens", {"default": 1024}),
    ("router", {"strategy": "length", "threshold": 40, "strong": STRONG, "weak": WEAK}),
    ("model_policy", {"allow": {"globex": ["mistral.ministral-*"], "*": ["*"]}}),
    ("budget", {"daily_usd": {"*": 0.01}}),
    ("guardrail", {"guardrail_id": "g"}),
    ("metering", {}),
]

failures = []


def check(name, cond):
    print(("ok  " if cond else "FAIL") + "  " + name)
    if not cond:
        failures.append(name)


def call(tenant="acme", text="hi", model="auto", max_tokens=None):
    body = {"model": model, "messages": [{"role": "user", "content": text}]}
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    return P.Call(body=body, headers={"x-tenant-id": tenant} if tenant else {})


S.set_store(FakeStore())
G._client = FakeBedrock()
p = pipe(*CHAIN)

c = call(text="short")
rej, tr = p.run_request(c)
check("short prompt routes weak", rej is None and c.model == f"mantle/{WEAK}")
check("max_tokens defaulted to cap", c.body["max_tokens"] == 1024)
check("every plugin timed", set(tr.timings_ms) == {n for n, _ in CHAIN})

c = call(text="x" * 60, max_tokens=32000)
p.run_request(c)
check("long prompt routes strong", c.model == f"mantle/{STRONG}")
check("oversized max_tokens clamped", c.body["max_tokens"] == 1024
      and c.attrs["max_tokens_clamped_from"] == 32000)

rej, tr = p.run_request(call(tenant="initech"))
check("unknown tenant rejected 403", rej and rej.status == 403 and tr.rejected_by == "tenant")

rej, tr = p.run_request(call(tenant="globex", text="x" * 60))
check("tenant barred from strong model", rej and rej.code == "model_not_allowed")

rej, tr = p.run_request(call(text="my ssn is 123"))
check("guardrail blocks", rej and rej.code == "guardrail_intervened" and rej.status == 400)
check("guardrail ran last among checks, after budget", list(tr.timings_ms)[-1] == "guardrail")

# spend: settle a response that costs more than the $0.01 budget, then get refused
c = call(text="x" * 60)
c.request_id = "r1"
p.run_request(c)
c.response = {"usage": {"prompt_tokens": 20000, "completion_tokens": 20000}}
tr = p.run_response(c)
check("response unwinds in reverse: metering before budget",
      list(tr.timings_ms).index("metering") < list(tr.timings_ms).index("budget"))
check("metering computed cost", c.attrs.get("cost_usd", 0) > 0.01)
check("budget settled the call", c.attrs.get("budget_settled") is True)
before = S.store().spend_today("acme")
p.run_response(c)
check("retried response does not bill twice", S.store().spend_today("acme") == before
      and c.attrs.get("budget_settled") is False)
rej, tr = p.run_request(call(text="again"))
check("budget exhausted -> 429", rej and rej.status == 429 and tr.rejected_by == "budget")
check("rejected before the guardrail was paid for", "guardrail" not in tr.timings_ms)

# streams: metering asks for usage on the way in
S.set_store(FakeStore())
c = call(text="stream me", max_tokens=500); c.body["stream"] = True
p.run_request(c)
check("streamed request asks the provider for usage", c.body.get("stream_options", {}).get("include_usage") is True)

# cache: a hit answers the call without a model
S.set_store(FakeStore())
cp = pipe(("tenant", {}), ("cache", {}), ("router", {"strategy": "pinned", "model": WEAK}), ("metering", {}))
c = call(text="what is 2+2")
v, tr = cp.run_request(c)
check("cache miss forwards", v is None and c.attrs["cache"] == "miss")
c.response = {"choices": [{"message": {"content": "4"}, "finish_reason": "stop"}],
              "usage": {"prompt_tokens": 5, "completion_tokens": 1}}
cp.run_response(c)
c2 = call(text="what is 2+2")
v2, tr2 = cp.run_request(c2)
check("cache hit answers from the interceptor", isinstance(v2, P.Serve)
      and v2.body["choices"][0]["message"]["content"] == "4" and v2.body["x_gateway"]["cached"] is True)
check("cache hit ends the chain before the router", "router" not in tr2.timings_ms)
c3 = call(text="what is 2+2"); c3.body["stream"] = True
v3, _ = cp.run_request(c3)
check("streamed requests bypass the cache", v3 is None)

# escalation: a weak answer that hit the ceiling is replaced by the strong model
import gateway.plugins.escalate as E  # noqa: E402
S.set_store(FakeStore())
E._mantle = lambda model, messages, max_tokens, region: {
    "model": model, "choices": [{"message": {"content": "a better, complete answer"},
                                 "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 20, "completion_tokens": 120}}
ep = pipe(("tenant", {}), ("router", {"strategy": "pinned", "model": WEAK}),
          ("metering", {}), ("escalate", {"strong": STRONG, "verbose_tokens": 400}))
c = call(text="explain the ocean"); c.request_id = "e1"
ep.run_request(c)
S.store().remember("e1", {"tenant": "acme", "model": f"mantle/{WEAK}", **c.attrs.get("remember", {})})
check("request phase remembered the prompt for a second call",
      "messages" in S.store().recall("e1"))
c.response = {"model": WEAK, "choices": [{"message": {"content": "half an ans"}, "finish_reason": "length"}],
              "usage": {"prompt_tokens": 20, "completion_tokens": 8}}
ep.run_response(c)
check("truncated weak answer is escalated", c.attrs.get("escalated_to") == STRONG
      and c.attrs["escalation_reason"] == "truncated")
check("replacement body is the strong model's answer",
      c.attrs["replacement_body"]["choices"][0]["message"]["content"].startswith("a better"))
check("both calls are billed", c.attrs["cost_usd"] >
      __import__("gateway.prices", fromlist=["x"]).cost_usd(STRONG, 20, 120))

c = call(text="explain the ocean"); c.request_id = "e2"
ep.run_request(c)
S.store().remember("e2", {"tenant": "acme", "model": f"mantle/{WEAK}", **c.attrs.get("remember", {})})
c.response = {"model": WEAK, "choices": [{"message": {"content": "a fine short answer"}, "finish_reason": "stop"}],
              "usage": {"prompt_tokens": 20, "completion_tokens": 12}}
ep.run_response(c)
check("a good weak answer is left alone", "replacement_body" not in c.attrs
      and c.attrs["escalation_reason"] == "none")

# a broken plugin: fail-open continues, fail-closed refuses
class Boom(P.Plugin):
    name = "boom"
    def on_request(self, call): raise RuntimeError("bug")
P.REGISTRY["boom"] = Boom
S.set_store(FakeStore())
ok = P.Pipeline.from_config({"plugins": [{"plugin": "boom"}, {"plugin": "tenant"}]})
rej, tr = ok.run_request(call())
check("fail-open: plugin error recorded, request continues", rej is None and "boom" in tr.errors)
closed = P.Pipeline.from_config({"fail_open": False, "plugins": [{"plugin": "boom"}]})
rej, _ = closed.run_request(call())
check("fail-closed: plugin error refuses with 503", rej and rej.status == 503)

w = pipe(("router", {"strategy": "weighted", "models": {WEAK: 50, STRONG: 50}}))
picks = set()
for i in range(40):
    c = call(text=f"prompt {i}"); w.run_request(c); picks.add(c.model)
c1, c2 = call(text="same"), call(text="same")
w.run_request(c1); w.run_request(c2)
check("weighted split uses both arms", len(picks) == 2)
check("weighted split is sticky per prompt", c1.model == c2.model)

c = call(model=f"mantle/{WEAK}")
pipe(("router", {"strategy": "pinned", "model": STRONG})).run_request(c)
check("explicit model passes through the router", c.model == f"mantle/{WEAK}")

# ---------------------------------------------------------------- handler contract
import base64, json, os  # noqa: E401,E402
from types import SimpleNamespace  # noqa: E402
from gateway import handler as H, pipeline as PP  # noqa: E402

os.environ["PIPELINE_CONFIG"] = json.dumps({"plugins": [{"plugin": n, "params": p} for n, p in CHAIN]})
PP._cache.update(at=0.0, pipeline=None, raw=None)
H.TABLE = "t"
S.set_store(FakeStore())
enc = lambda o: base64.b64encode(json.dumps(o).encode()).decode()  # noqa: E731
ctx = lambda rid: SimpleNamespace(client_context=SimpleNamespace(custom={"REQUEST_ID": rid}),  # noqa: E731
                                  aws_request_id="lambda-" + rid)

req_event = {"interceptorInputVersion": "1.0", "http": {"gatewayRequest": {
    "path": "/inference/v1/chat/completions", "httpMethod": "POST",
    "headers": {"X-Tenant-Id": "acme", "Authorization": "secret"},
    "body": enc({"model": "auto", "messages": [{"role": "user", "content": "hello"}]})}}}
out = H.lambda_handler(req_event, ctx("abc"))
fwd = json.loads(base64.b64decode(out["http"]["transformedGatewayRequest"]["body"]))
check("handler forwards a rewritten body", fwd["model"] == f"mantle/{WEAK}"
      and out["interceptorOutputVersion"] == "1.0")
check("request phase remembered the tenant for the response", S.store().recall("abc").get("tenant") == "acme")

resp_event = {"interceptorInputVersion": "1.0", "http": {"gatewayRequest": None, "gatewayResponse": {
    "statusCode": 200, "headers": None, "contentType": "application/json",
    "body": enc({"model": WEAK, "usage": {"prompt_tokens": 12, "completion_tokens": 40}})}}}
out = H.lambda_handler(resp_event, ctx("abc"))
annotated = json.loads(base64.b64decode(out["http"]["transformedGatewayResponse"]["body"]))
check("response phase recovers tenant and annotates cost", annotated["x_gateway"]["tenant"] == "acme"
      and annotated["x_gateway"]["cost_usd"] > 0 and S.store().spend_today("acme") > 0)
check("provider fields preserved in annotated body", annotated["usage"]["completion_tokens"] == 40)
check("response for a rejected request is a no-op", H.lambda_handler(resp_event, ctx("never-forwarded")) == H.PASS)

bad = json.loads(json.dumps(req_event)); bad["http"]["gatewayRequest"]["headers"]["X-Tenant-Id"] = "initech"
out = H.lambda_handler(bad, ctx("def"))
short = out["http"]["transformedGatewayResponse"]
check("rejection short-circuits with status and error body", short["statusCode"] == 403
      and json.loads(base64.b64decode(short["body"]))["error"]["code"] == "unknown_tenant"
      and "transformedGatewayRequest" not in out["http"])
sse = ("data: " + json.dumps({"model": WEAK, "choices": [{"delta": {"content": "hi"}}]}) + "\n\n"
       "data: " + json.dumps({"model": WEAK, "choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 90}}) + "\n\n"
       "data: [DONE]\n\n")
S.store().remember("strm", {"tenant": "acme", "model": f"mantle/{WEAK}"})
before = S.store().spend_today("acme")
sse_event = {"http": {"gatewayRequest": None, "gatewayResponse": {"statusCode": 200,
             "contentType": "text/event-stream", "body": base64.b64encode(sse.encode()).decode()}}}
out = H.lambda_handler(sse_event, ctx("strm"))
check("buffered stream metered from its final chunk and settled", S.store().spend_today("acme") > before)
check("event stream passed through unmodified", out == H.PASS)
check("handler never raises on garbage", H.lambda_handler({"http": {"gatewayRequest": {"body": "!!"}}}, ctx("x"))
      == H.PASS)

print(f"\n{'FAILED' if failures else 'all passed'}: {len(failures)} failure(s)")
sys.exit(1 if failures else 0)
