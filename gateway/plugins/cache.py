"""Answer repeat requests from a cache, without calling a model.

The same mechanism that refuses a call can also answer one: a REQUEST
interceptor that returns a response short-circuits the gateway, so a cache hit
never reaches Bedrock and costs no tokens. AgentCore has no native response
cache, which makes this the single largest cost lever available in the
interceptor.

Keyed on the request as the client sent it -- the virtual model, the messages
and the sampling parameters -- so the entry is "the answer we would serve for
this request", including one an escalation later improved. Place it before the
router for that reason.

Exact matching only. Semantic caching needs an embedding per request, which
costs a round trip and turns a 5 ms plugin into a 110 ms one; the cheap version
earns its place first.
"""
import hashlib
import json
import os
import time

from ..pipeline import Call, Plugin, Serve, register
from ..state import store


@register
class Cache(Plugin):
    name = "cache"
    needs_response = True

    def _key(self, call: Call) -> str:
        material = json.dumps({
            "model": call.model,
            "messages": call.body.get("messages") or call.body.get("input"),
            "temperature": call.body.get("temperature"),
            "max_tokens": call.body.get("max_tokens"),
            "tenant": call.tenant if self.params.get("per_tenant", True) else "*",
        }, sort_keys=True, default=str)
        return hashlib.sha256(material.encode()).hexdigest()[:32]

    def on_request(self, call: Call) -> Serve | None:
        if call.body.get("stream"):
            return None                      # a cached body is not an event stream
        key = self._key(call)
        call.attrs["cache_key"] = key
        hit = store(self.params.get("table") or os.environ.get("STATE_TABLE")).get_cached(key)
        if hit is None:
            call.attrs["cache"] = "miss"
            return None
        call.attrs["cache"] = "hit"
        hit["x_gateway"] = {"cached": True, "tenant": call.tenant}
        return Serve(body=hit)

    def on_response(self, call: Call) -> None:
        key = call.attrs.get("cache_key")
        body = call.attrs.get("replacement_body") or call.response
        if not key or not body or call.attrs.get("streamed") or not body.get("choices"):
            return
        store(self.params.get("table") or os.environ.get("STATE_TABLE")).put_cached(
            key, body, ttl_s=int(self.params.get("ttl_s", 3600)))
        call.attrs["cached_at"] = int(time.time())
