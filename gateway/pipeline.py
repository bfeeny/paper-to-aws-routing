"""A plugin pipeline for AgentCore Gateway interceptors.

An interceptor is a single Lambda invoked before (REQUEST) and after (RESPONSE)
every inference call. Everything an operator might want to do at that moment --
check a budget, clamp max_tokens, run a guardrail, pick the model, meter the
cost -- is a small, independent decision about the same request. Writing them
as one function couples them; writing them as plugins lets each be added,
removed, reordered or swapped without touching the others, and lets the order
be changed at runtime from configuration rather than by redeploying.

Contract:

  * A plugin sees a `Call` -- the parsed request body, headers, the resolved
    tenant, and a scratch dict (`attrs`) that plugins use to hand facts to
    later plugins (a price estimate, a routing score).
  * `on_request` may mutate the body, or end the call early by returning
    `Reject` (an error) or `Serve` (a ready-made answer, as the cache does).
    The first plugin to do either wins; no later plugin runs.
  * `on_response` sees the provider's response and may record, rewrite or
    annotate it. It cannot reject: by then the tokens are already spent.
  * `on_abort` is the path back out of an early exit. A plugin that *reserved*
    something -- quota, a slot, a lock -- has to release it when a later plugin
    ends the call, because the reservation was for a model call that is now
    never going to happen. Plugins that only observe do not need it.
  * Responses unwind the chain in reverse, as middleware does: the plugin that
    ran last on the way in runs first on the way out. Metering sits at the end
    of the request chain so that it has priced the call before the budget
    plugin, earlier in the chain, settles it.
  * Every plugin is timed. One EMF record per call carries the per-plugin
    latency, which is how the overhead of each customization is measured.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Call:
    body: dict
    headers: dict = field(default_factory=dict)
    path: str = ""
    tenant: str = "anonymous"
    request_id: str = ""                  # the gateway's REQUEST_ID, shared by both phases
    attrs: dict = field(default_factory=dict)
    response: dict | None = None          # set for the RESPONSE phase

    @property
    def model(self) -> str:
        return str(self.body.get("model", ""))

    def prompt_text(self) -> str:
        """User-visible text, across chat-completions, messages and responses shapes."""
        b = self.body
        if isinstance(b.get("messages"), list):
            out = []
            for m in b["messages"]:
                c = m.get("content")
                if isinstance(c, str):
                    out.append(c)
                elif isinstance(c, list):
                    out.extend(x.get("text", "") for x in c if isinstance(x, dict))
            return "\n".join(out)
        inp = b.get("input", "")
        return inp if isinstance(inp, str) else json.dumps(inp)


@dataclass
class Reject:
    status: int
    code: str
    message: str

    def as_body(self) -> dict:
        # OpenAI-compatible error envelope, so existing client SDKs surface it cleanly.
        return {"error": {"type": self.code, "code": self.code, "message": self.message}}


@dataclass
class Serve:
    """Answer the call from the interceptor, without calling a model."""

    body: dict
    status: int = 200


class Plugin:
    """Base class. Override `on_request` and/or `on_response`."""

    name = "plugin"
    needs_response = False   # True if on_response needs state from the request phase

    def __init__(self, **params):
        self.params = params

    def on_request(self, call: Call) -> Reject | Serve | None:  # noqa: ARG002
        return None

    def on_response(self, call: Call) -> None:  # noqa: ARG002
        return None

    def on_abort(self, call: Call, verdict) -> None:  # noqa: ARG002
        """Called on plugins that already ran when a later one ends the call."""
        return None


REGISTRY: dict[str, type[Plugin]] = {}


def register(cls: type[Plugin]) -> type[Plugin]:
    REGISTRY[cls.name] = cls
    return cls


@dataclass
class Trace:
    phase: str
    timings_ms: dict = field(default_factory=dict)
    rejected_by: str | None = None
    errors: dict = field(default_factory=dict)


class Pipeline:
    def __init__(self, plugins: list[Plugin], fail_open: bool = True):
        self.plugins = plugins
        # A plugin that raises is a bug in the plugin, not a verdict on the
        # request. Default is to log it and continue; set fail_open=false for
        # plugins whose absence is unacceptable (a compliance guardrail).
        self.fail_open = fail_open

    @classmethod
    def from_config(cls, cfg: dict) -> "Pipeline":
        plugins = []
        for spec in cfg.get("plugins", []):
            name = spec["plugin"]
            if name not in REGISTRY:
                raise ValueError(f"unknown plugin {name!r}; registered: {sorted(REGISTRY)}")
            plugins.append(REGISTRY[name](**spec.get("params", {})))
        return cls(plugins, fail_open=cfg.get("fail_open", True))

    def run_request(self, call: Call) -> tuple[Reject | Serve | None, Trace]:
        return self._run("request", call, lambda p: p.on_request(call))

    def run_response(self, call: Call) -> Trace:
        return self._run("response", call, lambda p: p.on_response(call))[1]

    def _run(self, phase: str, call: Call, step: Callable[[Plugin], Reject | None]):
        trace = Trace(phase)
        order = self.plugins if phase == "request" else list(reversed(self.plugins))
        for plugin in order:
            t0 = time.perf_counter()
            try:
                verdict = step(plugin)
            except Exception as exc:  # noqa: BLE001 - recorded, and the policy decides
                trace.errors[plugin.name] = repr(exc)[:300]
                trace.timings_ms[plugin.name] = (time.perf_counter() - t0) * 1000
                if not self.fail_open:
                    return Reject(503, "plugin_error",
                                  f"{plugin.name} failed and the pipeline is fail-closed"), trace
                continue
            trace.timings_ms[plugin.name] = (time.perf_counter() - t0) * 1000
            if phase == "request" and isinstance(verdict, (Reject, Serve)):
                trace.rejected_by = plugin.name
                self._abort(order, plugin, call, verdict, trace)
                return verdict, trace
        return None, trace

    def _abort(self, order, stopper, call: Call, verdict, trace: Trace) -> None:
        """Unwind the plugins that already ran, newest first, as a stack would.

        The plugin that ended the call is skipped: it knows what it did. An
        exception here is logged and ignored -- a failed release must not turn
        a clean rejection into a 500.
        """
        ran = order[:order.index(stopper)]
        for plugin in reversed(ran):
            try:
                plugin.on_abort(call, verdict)
            except Exception as exc:  # noqa: BLE001
                trace.errors[f"{plugin.name}.abort"] = repr(exc)[:200]


# ---------------------------------------------------------------- configuration
#
# The plugin list lives in SSM Parameter Store so it can change without a
# deploy. It is cached per Lambda container for CONFIG_TTL seconds: long enough
# that a busy gateway does not call SSM per request, short enough that a change
# propagates within a minute.

_cache: dict = {"at": 0.0, "pipeline": None, "raw": None}


def load_pipeline(ssm_client=None) -> Pipeline:
    ttl = float(os.environ.get("CONFIG_TTL", "30"))
    if _cache["pipeline"] is not None and time.time() - _cache["at"] < ttl:
        return _cache["pipeline"]
    raw = os.environ.get("PIPELINE_CONFIG")
    param = os.environ.get("PIPELINE_CONFIG_PARAM")
    if param:
        if ssm_client is None:
            import boto3
            ssm_client = boto3.client("ssm")
        raw = ssm_client.get_parameter(Name=param)["Parameter"]["Value"]
    if raw != _cache["raw"] or _cache["pipeline"] is None:
        _cache["pipeline"] = Pipeline.from_config(json.loads(raw or '{"plugins": []}'))
        _cache["raw"] = raw
    _cache["at"] = time.time()
    return _cache["pipeline"]


# ---------------------------------------------------------------- observability

def emit(trace: Trace, call: Call, namespace: str, outcome: str) -> None:
    """One EMF record per phase: per-plugin latency plus the decision fields."""
    metrics = [{"Name": f"{k}Ms", "Unit": "Milliseconds"} for k in trace.timings_ms]
    metrics.append({"Name": "PipelineMs", "Unit": "Milliseconds"})
    record = {
        "_aws": {
            "Timestamp": int(time.time() * 1000),
            "CloudWatchMetrics": [{"Namespace": namespace,
                                   "Dimensions": [["phase"], ["phase", "outcome"]],
                                   "Metrics": metrics}],
        },
        "phase": trace.phase,
        "outcome": outcome,
        "tenant": call.tenant,
        "model": call.model,
        "ended_by": trace.rejected_by,
        "errors": trace.errors or None,
        "attrs": {k: v for k, v in call.attrs.items() if isinstance(v, (str, int, float, bool))},
        "PipelineMs": round(sum(trace.timings_ms.values()), 3),
        **{f"{k}Ms": round(v, 3) for k, v in trace.timings_ms.items()},
    }
    print(json.dumps(record, default=str))
