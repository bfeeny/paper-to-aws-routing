"""Which models may this tenant use?

Runs after the router, so it judges the model that will actually be called,
not the virtual alias the client sent. Patterns use shell-style globs, so
`mistral.*` admits a family and `*` admits everything.
"""
import fnmatch

from ..pipeline import Call, Plugin, Reject, register
from ..prices import bare


@register
class ModelPolicy(Plugin):
    name = "model_policy"

    def on_request(self, call: Call) -> Reject | None:
        allow = self.params.get("allow", {})
        patterns = allow.get(call.tenant, allow.get("*", ["*"]))
        model = bare(call.model)
        if any(fnmatch.fnmatch(model, p) for p in patterns):
            return None
        return Reject(403, "model_not_allowed",
                      f"tenant {call.tenant!r} may not call {model!r}")
