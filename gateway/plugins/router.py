"""Resolve a virtual model ID to a concrete model.

The gateway routes on the `model` field, so a router is any function that
rewrites it before routing. Callers send a stable alias (`auto`); the strategy
behind it can change without a client release. Requests naming a real model
pass through untouched.

Strategies:
  pinned    always one model -- the null router, and the right default
  length    prompt length threshold (the AgentCore interceptor docs example)
  weighted  sticky percentage split between models, for A/B tests and canaries
  learned   a trained scorer against a calibrated threshold
"""
import hashlib

from ..pipeline import Call, Plugin, register


@register
class Router(Plugin):
    name = "router"
    _scorer = None

    def on_request(self, call: Call):
        virtual = self.params.get("virtual", "auto")
        if call.model != virtual:
            return None
        strategy = self.params.get("strategy", "pinned")
        model = getattr(self, f"_{strategy}")(call)
        call.attrs["route_strategy"] = strategy
        target = self.params.get("target", "mantle")
        call.body["model"] = model if "/" in model else f"{target}/{model}"
        return None

    def _pinned(self, call: Call) -> str:
        return self.params["model"]

    def _length(self, call: Call) -> str:
        n = len(call.prompt_text())
        call.attrs["route_score"] = n
        return self.params["strong"] if n >= int(self.params.get("threshold", 2000)) \
            else self.params["weak"]

    def _weighted(self, call: Call) -> str:
        # Hash the prompt, not a random draw: the same request lands on the same
        # arm on retry, which keeps an A/B comparison from double-counting.
        models = self.params["models"]            # {"model-a": 90, "model-b": 10}
        total = sum(models.values())
        h = int(hashlib.sha256(call.prompt_text().encode()).hexdigest()[:8], 16) % total
        for m, w in models.items():
            if h < w:
                return m
            h -= w
        return next(iter(models))

    def _learned(self, call: Call) -> str:
        if Router._scorer is None:
            from ..routellm_scorer import load
            Router._scorer = load()
        score = float(Router._scorer(call.prompt_text()))
        call.attrs["route_score"] = round(score, 4)
        cut = self.params.get("threshold")
        if cut is None:
            from ..routellm_scorer import threshold_for
            cut = threshold_for(int(self.params.get("call_rate_pct", 30)))
        return self.params["strong"] if score >= float(cut) else self.params["weak"]
