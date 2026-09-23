#!/usr/bin/env python3
"""Offline tests for the gateway plugin pipeline. No AWS calls: clients are faked.

    python3 tests/test_pipeline.py
"""
import json
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
    def __init__(self): self.spend, self.reqs, self.settled, self.cache = {}, {}, set(), {}; self.vectors = []; self.windows = {}
    # Serializes on write, as the real store does: returning the caller's own
    # dict would let a later mutation appear to have been cached.
    def get_cached(self, key):
        return json.loads(self.cache[key]) if key in self.cache else None
    def put_cached(self, key, body, ttl_s=0): self.cache[key] = json.dumps(body)
    def spend_today(self, tenant): return self.spend.get(tenant, 0.0)
    def remember(self, rid, data): self.reqs[rid] = json.loads(json.dumps(data))
    def recall(self, rid): return self.reqs.get(rid, {})
    def bump_window(self, tenant, window, window_s, requests=1, tokens=0, smooth=False):
        c = self.windows.setdefault((tenant, window), {"requests": 0.0, "tokens": 0.0})
        c["requests"] += requests; c["tokens"] += tokens
        return dict(c)
    def put_vector(self, key, tenant, vec, prompt, ttl_s=0, index_key=None):
        self.vectors.append({"key": key, "tenant": tenant, "vec": list(vec), "prompt": prompt})
    def nearest(self, index, tenant, vec, top_k=1):
        def cos(a, b):
            num = sum(x * y for x, y in zip(a, b))
            den = (sum(x * x for x in a) ** 0.5) * (sum(y * y for y in b) ** 0.5) or 1.0
            return num / den
        hits = [{"similarity": cos(vec, v["vec"]), "cache_key": v["key"], "prompt": v["prompt"]}
                for v in self.vectors if v["tenant"] == tenant]
        return sorted(hits, key=lambda h: -h["similarity"])[:top_k]
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
check("cache key is remembered for the response phase", "cache_key" in c.attrs.get("remember", {}))
resp_call = P.Call(body={"model": WEAK}, tenant=c.tenant, request_id=c.request_id,
                   response={"choices": [{"message": {"content": "4"}, "finish_reason": "stop"}],
                             "usage": {"prompt_tokens": 5, "completion_tokens": 1}})
# the handler hands the response phase only what the request phase remembered
resp_call.attrs["recalled"] = c.attrs.get("remember", {})
resp_call.response = resp_call.response
cp.run_response(resp_call)
c2 = call(text="what is 2+2")
v2, tr2 = cp.run_request(c2)
check("cache hit answers from the interceptor", isinstance(v2, P.Serve)
      and v2.body["choices"][0]["message"]["content"] == "4" and v2.body["x_gateway"]["cached"] is True)
check("cache hit ends the chain before the router", "router" not in tr2.timings_ms)
c3 = call(text="what is 2+2"); c3.body["stream"] = True
v3, _ = cp.run_request(c3)
check("streamed requests bypass the cache", v3 is None)

# semantic cache: similarity recalls a candidate, a cheap model decides whether to serve it
import gateway.plugins.semantic_cache as SC  # noqa: E402

S.set_store(FakeStore())
VECTORS = {                                   # stand-in embeddings, hand-placed
    "what is the capital of france": [1.0, 0.0, 0.0],
    "which city is the capital of france":  [0.97, 0.24, 0.0],   # paraphrase, same answer
    "what is the capital of finland": [0.99, 0.10, 0.0],         # near-identical text, other answer
    "explain tail latency": [0.0, 0.0, 1.0],                     # unrelated
}
SC._embed = lambda text, dims=256: VECTORS[text.strip().lower()]
verifier_calls = []
def fake_equivalent(model, a, b):
    verifier_calls.append((a, b))
    return {"france": "france", "finland": "finland"}.get(
        a.split()[-1].lower()) == b.split()[-1].lower()
SC._equivalent = fake_equivalent

sp = pipe(("tenant", {}), ("semantic_cache", {"threshold": 0.80}),
          ("router", {"strategy": "pinned", "model": WEAK}), ("metering", {}))

def seed(text, answer):
    c = call(text=text)
    sp.run_request(c)
    r = P.Call(body={"model": WEAK}, tenant=c.tenant, request_id=c.request_id,
               response={"choices": [{"message": {"content": answer}, "finish_reason": "stop"}],
                         "usage": {"prompt_tokens": 5, "completion_tokens": 1}})
    r.attrs["recalled"] = c.attrs.get("remember", {})
    sp.run_response(r)
    return c

first = seed("what is the capital of France", "Paris")
check("first semantic request is a miss and stores nothing to serve", first.attrs["semantic_cache"] == "empty")
check("the vector and prompt are remembered for the response phase",
      {"sem_vec", "prompt"} <= set(first.attrs.get("remember", {})))
check("the remembered vector is not named `embedding` -- that name is the "
      "table's vector attribute and is reserved for every item in it",
      "embedding" not in first.attrs.get("remember", {}))
check("the response phase stores one vector", len(S.store().vectors) == 1)

verifier_calls.clear()
c = call(text="which city is the capital of France")
v, tr = sp.run_request(c)
check("a paraphrase is recalled and served", isinstance(v, P.Serve)
      and v.body["choices"][0]["message"]["content"] == "Paris"
      and v.body["x_gateway"]["semantic"] is True)
check("a semantic hit ends the chain before the router", "router" not in tr.timings_ms)
check("the verifier was asked exactly once", len(verifier_calls) == 1)

verifier_calls.clear()
c = call(text="what is the capital of Finland")
v, _ = sp.run_request(c)
check("a closer neighbour with a different answer is recalled but refused",
      v is None and c.attrs["semantic_cache"] == "rejected_by_verifier")
check("the refused neighbour was nearer than the served paraphrase",
      c.attrs["semantic_similarity"] > 0.95 and len(verifier_calls) == 1)

c = call(text="explain tail latency")
v, _ = sp.run_request(c)
check("an unrelated prompt never reaches the verifier",
      v is None and c.attrs["semantic_cache"] == "below_threshold"
      and c.attrs["semantic_verifier_calls"] == 0)

S.store().cache.clear()                       # the answer expires, its vector does not
c = call(text="which city is the capital of France")
v, _ = sp.run_request(c)
check("a vector outliving its answer is a miss, not a crash", v is None)

nv = pipe(("tenant", {}), ("semantic_cache", {"threshold": 0.80, "verify": False}))
S.set_store(FakeStore()); seed("what is the capital of France", "Paris")
verifier_calls.clear()
c = call(text="what is the capital of Finland")
v, _ = nv.run_request(c)
check("verify=false serves the wrong answer -- which is why it defaults on",
      isinstance(v, P.Serve) and not verifier_calls)

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

# a client asking for a short answer must not trigger escalation
ep2 = pipe(("tenant", {}), ("max_tokens", {"default": 1024}),
           ("router", {"strategy": "pinned", "model": WEAK}), ("metering", {}),
           ("escalate", {"strong": STRONG}))
c = call(text="say hi", max_tokens=16); c.request_id = "e3"
ep2.run_request(c)
check("client-set max_tokens is recorded as the client's", c.attrs["max_tokens_source"] == "client")
S.store().remember("e3", {"tenant": "acme", "model": f"mantle/{WEAK}", **c.attrs.get("remember", {})})
c.attrs["recalled"] = S.store().recall("e3")
c.response = {"model": WEAK, "choices": [{"message": {"content": "Hi there, I am"}, "finish_reason": "length"}],
              "usage": {"prompt_tokens": 5, "completion_tokens": 16}}
ep2.run_response(c)
check("truncation from the client's own cap does not escalate", "replacement_body" not in c.attrs)

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

# rate limits: reserve max_tokens up front, reconcile with real usage on the way out
import gateway.plugins.rate_limit as RL  # noqa: E402

S.set_store(FakeStore())
rp = pipe(("tenant", {}), ("rate_limit", {"requests_per_min": {"*": 3}, "tokens_per_min": {"*": 1000}}))
v, _ = rp.run_request(call(text="hi", max_tokens=100))
check("first request passes and counts itself", v is None)
for _ in range(2):
    rp.run_request(call(text="hi", max_tokens=100))
v, _ = rp.run_request(call(text="hi", max_tokens=100))
check("the request over the limit is the one refused",
      isinstance(v, P.Reject) and v.status == 429 and v.code == "rate_limit_requests")

S.set_store(FakeStore())
rp = pipe(("tenant", {}), ("rate_limit", {"tokens_per_min": {"*": 1000}}))
v, _ = rp.run_request(call(text="hi", max_tokens=900))
check("a big reservation passes while it fits", v is None)
c2 = call(text="hi", max_tokens=900)
v, _ = rp.run_request(c2)
check("two reservations that cannot both fit are caught before either answers",
      isinstance(v, P.Reject) and v.code == "rate_limit_tokens")
check("the refusal names the reservation, not the usage", "reserved 900" in v.message)

S.set_store(FakeStore())
rp = pipe(("tenant", {}), ("rate_limit", {"tokens_per_min": {"*": 1000}}), ("metering", {}))
c = call(text="hi", max_tokens=900)
rp.run_request(c)
resp = P.Call(body={"model": WEAK}, tenant=c.tenant, request_id=c.request_id,
              response={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 5, "completion_tokens": 12}})
resp.attrs["recalled"] = c.attrs.get("remember", {})
rp.run_response(resp)
after, _ = pipe(("tenant", {}), ("rate_limit", {"tokens_per_min": {"*": 1000}})).run_request(
    call(text="hi", max_tokens=900))
check("the unused part of a reservation is given back", after is None)

# a reservation must be released when a later plugin ends the call
S.set_store(FakeStore())
ap = pipe(("tenant", {}), ("rate_limit", {"tokens_per_min": {"*": 5000}}), ("cache", {}),
          ("metering", {}))
c = call(text="what is 2+2", max_tokens=4000)
ap.run_request(c)
resp = P.Call(body={"model": WEAK}, tenant=c.tenant, request_id=c.request_id,
              response={"choices": [{"message": {"content": "4"}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 5, "completion_tokens": 1}})
resp.attrs["recalled"] = c.attrs.get("remember", {})
ap.run_response(resp)
hit = call(text="what is 2+2", max_tokens=4000)
v, _ = ap.run_request(hit)
check("the cached answer is served", isinstance(v, P.Serve))
check("a cache hit releases the quota it reserved", hit.attrs.get("rl_released") == 4000)
window = S.store().windows[(hit.tenant, next(iter(k[1] for k in S.store().windows)))]
check("a cache hit leaves the token window where it found it", round(window["tokens"]) == 6)
check("but the request still counts as a request", window["requests"] == 2)

S.set_store(FakeStore())
bp = pipe(("tenant", {}), ("rate_limit", {"tokens_per_min": {"*": 5000}}),
          ("budget", {"daily_usd": {"*": 0}}))
c = call(text="hi", max_tokens=4000)
v, _ = bp.run_request(c)
check("a budget rejection also releases the reservation",
      isinstance(v, P.Reject) and c.attrs.get("rl_released") == 4000)

# PII: mask on the way in, restore on the way out
import gateway.plugins.pii as PII  # noqa: E402

spans = PII._spans_regex("mail bob@acme.com or bob@acme.com, ssn 123-45-6789", None)
masked, mapping = PII.mask("mail bob@acme.com or bob@acme.com, ssn 123-45-6789", spans)
check("the same value gets the same placeholder twice", masked.count("{EMAIL_0}") == 2)
check("distinct types get distinct placeholders", "{SSN_0}" in masked and len(mapping) == 2)
check("no original value survives masking", "bob@acme.com" not in masked and "123-45-6789" not in masked)
check("unmasking is exact", PII.unmask(masked, mapping)
      == "mail bob@acme.com or bob@acme.com, ssn 123-45-6789")

S.set_store(FakeStore())
pp = pipe(("tenant", {}), ("pii", {"detector": "regex"}))
c = call(text="email bob@acme.com about invoice 7")
pp.run_request(c)
sent = c.body["messages"][0]["content"]
check("the model never sees the address", "bob@acme.com" not in sent and "{EMAIL_0}" in sent)
resp = P.Call(body={"model": WEAK}, tenant=c.tenant, request_id=c.request_id,
              response={"choices": [{"message": {"content": "I emailed {EMAIL_0} about it."},
                                     "finish_reason": "stop"}]})
resp.attrs["recalled"] = c.attrs.get("remember", {})
pp.run_response(resp)
check("the caller gets the real address back",
      resp.response["choices"][0]["message"]["content"] == "I emailed bob@acme.com about it.")

c = call(text="email bob@acme.com")
pipe(("tenant", {}), ("pii", {"detector": "regex", "restore": False})).run_request(c)
check("restore=false never persists the originals", "pii_map" not in c.attrs.get("remember", {}))

# pii before cache: nothing personal is stored, and a hit is re-personalized
S.set_store(FakeStore())
cp2 = pipe(("tenant", {}), ("pii", {"detector": "regex"}), ("cache", {}))
c = call(text="email bob@acme.com")
cp2.run_request(c)
resp = P.Call(body={"model": WEAK}, tenant=c.tenant, request_id=c.request_id,
              response={"choices": [{"message": {"content": "I mailed {EMAIL_0}."},
                                     "finish_reason": "stop"}]})
resp.attrs["recalled"] = c.attrs.get("remember", {})
cp2.run_response(resp)
stored = json.loads(S.store().cache[c.attrs["cache_key"]])["choices"][0]["message"]["content"]
check("the cache stores the de-identified answer", "{EMAIL_0}" in stored
      and "bob@acme.com" not in stored)
hit = call(text="email bob@acme.com")
v, _ = cp2.run_request(hit)
check("a cache hit is re-personalized before it is served", isinstance(v, P.Serve)
      and v.body["choices"][0]["message"]["content"] == "I mailed bob@acme.com.")

other = call(text="email carol@globex.com")
v2, _ = cp2.run_request(other)
check("a different person asking the same question shares the entry",
      isinstance(v2, P.Serve)
      and v2.body["choices"][0]["message"]["content"] == "I mailed carol@globex.com.")

# JWT tenancy: the tenant comes from a signed token, not a header
import gateway.plugins.jwt_tenant as JT  # noqa: E402
try:
    import jwt as _jwtlib
    from cryptography.hazmat.primitives.asymmetric import rsa
    HAVE_JWT = True
except ImportError:  # pragma: no cover - exercised only where PyJWT is absent
    HAVE_JWT = False

if HAVE_JWT:
    import time as _t
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    ISS = "https://issuer.example/pool"

    def token(claims, signer=key, alg="RS256"):
        return _jwtlib.encode({"iss": ISS, "exp": int(_t.time()) + 300, **claims},
                              signer, algorithm=alg, headers={"kid": "k1"})

    pub = _jwtlib.algorithms.RSAAlgorithm.to_jwk(key.public_key(), as_dict=True)
    pub["kid"] = "k1"
    JT._jwks.update(at=_t.time(), url=ISS + "/.well-known/jwks.json", keys={"k1": pub})

    jp = pipe(("jwt_tenant", {"issuer": ISS, "claim": "client_id",
                              "map": {"abc123": "acme"}, "known": ["acme"]}))
    c = call(text="hi"); c.headers["authorization"] = "Bearer " + token({"client_id": "abc123"})
    v, _ = jp.run_request(c)
    check("a verified token sets the tenant", v is None and c.tenant == "acme"
          and c.attrs["jwt"] == "verified")

    c = call(text="hi"); c.headers["authorization"] = "Bearer " + token({"client_id": "nope"})
    v, _ = jp.run_request(c)
    check("a valid token for an unprovisioned tenant is refused",
          isinstance(v, P.Reject) and v.code == "unknown_tenant")

    other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    c = call(text="hi")
    c.headers["authorization"] = "Bearer " + token({"client_id": "abc123"}, signer=other)
    v, _ = jp.run_request(c)
    check("a token signed by the wrong key is refused",
          isinstance(v, P.Reject) and v.status == 401 and v.code == "invalid_token")

    c = call(text="hi")
    c.headers["authorization"] = "Bearer " + _jwtlib.encode(
        {"iss": ISS, "exp": int(_t.time()) - 10, "client_id": "abc123"}, key,
        algorithm="RS256", headers={"kid": "k1"})
    v, _ = jp.run_request(c)
    check("an expired token is refused", isinstance(v, P.Reject) and v.code == "invalid_token")

    c = call(text="hi")
    v, _ = jp.run_request(c)
    check("no token at all is refused", isinstance(v, P.Reject) and v.code == "no_token")

    sp = pipe(("jwt_tenant", {"issuer": ISS, "scope_prefix": "tenant-"}))
    c = call(text="hi")
    c.headers["authorization"] = "Bearer " + token({"scope": "gw/tenant-globex"})
    v, _ = sp.run_request(c)
    check("tenancy can come from a resource-server scope", v is None and c.tenant == "globex")

    c = call(text="hi")
    c.headers["authorization"] = "Bearer " + token({"scope": "gw/tenant-globex",
                                                    "client_id": "abc123"})
    v, _ = sp.run_request(c)
    check("a scope outranks the client id when both are present",
          v is None and c.tenant == "globex")

    # The header the rest of the pipeline used to trust
    hp = pipe(("jwt_tenant", {"issuer": ISS, "claim": "client_id", "map": {"abc123": "acme"}}))
    c = call(text="hi")
    c.headers["x-tenant-id"] = "globex"
    c.headers["authorization"] = "Bearer " + token({"client_id": "abc123"})
    hp.run_request(c)
    check("a spoofed tenant header cannot override the token", c.tenant == "acme")
else:
    print("skip  JWT tests (PyJWT not installed)")

print(f"\n{'FAILED' if failures else 'all passed'}: {len(failures)} failure(s)")
sys.exit(1 if failures else 0)
