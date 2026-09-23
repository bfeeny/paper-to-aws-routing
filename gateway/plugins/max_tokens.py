"""Clamp the output-token request before it reaches the model.

Bedrock reserves input tokens plus the full max_tokens against a model's
tokens-per-minute quota when a request starts, and refunds the unused part
only when it finishes. Clients that ask for 32k "to be safe" throttle the whole
account. Capping here costs nothing and returns quota to everyone else.
"""
from ..pipeline import Call, Plugin, register
from ..prices import bare

FIELDS = ("max_tokens", "max_completion_tokens", "max_output_tokens")


@register
class MaxTokens(Plugin):
    name = "max_tokens"

    def on_request(self, call: Call):
        caps = self.params.get("per_model", {})
        cap = int(caps.get(bare(call.model), self.params.get("default", 2048)))
        present = [f for f in FIELDS if f in call.body]
        # Record who set the ceiling. Escalation must not treat a client's own
        # short answer as a failure -- see the escalate plugin.
        call.attrs["max_tokens_source"] = "client" if present else "gateway"
        call.attrs.setdefault("remember", {})["max_tokens_source"] = call.attrs["max_tokens_source"]
        for f in present or [self.params.get("field", "max_tokens")]:
            asked = call.body.get(f)
            new = cap if asked is None else min(int(asked), cap)
            if asked is not None and int(asked) > cap:
                call.attrs["max_tokens_clamped_from"] = int(asked)
            call.body[f] = new
        return None
