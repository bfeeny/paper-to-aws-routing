#!/usr/bin/env python3
"""Offline tests for the gateway plugin pipeline. No AWS calls: clients are faked.

    python3 tests/test_pipeline.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from gateway import pipeline as P  # noqa: E402
from gateway import plugins  # noqa: E402,F401  (registers plugins)
from gateway.plugins import budget as B, guardrail as G  # noqa: E402

WEAK, STRONG = "mistral.ministral-3-3b-instruct", "mistral.mistral-large-3-675b-instruct"


class FakeTable:
    def __init__(self): self.items = {}
    def get_item(self, Key): return {"Item": self.items[Key["pk"]]} if Key["pk"] in self.items else {}
    def update_item(self, Key, ExpressionAttributeValues, **_):
        it = self.items.setdefault(Key["pk"], {"spend_usd": 0})
        it["spend_usd"] = float(it["spend_usd"]) + float(ExpressionAttributeValues[":c"])


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
    ("budget", {"table": "t", "daily_usd": {"*": 0.01}}),
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


B._table = FakeTable()
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
p.run_request(c)
c.response = {"usage": {"prompt_tokens": 20000, "completion_tokens": 20000}}
p.run_response(c)
check("metering computed cost", c.attrs.get("cost_usd", 0) > 0.01)
rej, tr = p.run_request(call(text="again"))
check("budget exhausted -> 429", rej and rej.status == 429 and tr.rejected_by == "budget")
check("rejected before the guardrail was paid for", "guardrail" not in tr.timings_ms)

# a broken plugin: fail-open continues, fail-closed refuses
class Boom(P.Plugin):
    name = "boom"
    def on_request(self, call): raise RuntimeError("bug")
P.REGISTRY["boom"] = Boom
B._table = FakeTable()
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

print(f"\n{'FAILED' if failures else 'all passed'}: {len(failures)} failure(s)")
sys.exit(1 if failures else 0)
