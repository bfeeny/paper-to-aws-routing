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

Two backends, chosen by the CACHE_BACKEND environment variable:

  dynamodb  on-demand table, reachable from a Lambda outside any VPC. Nothing
            to size, nothing billed at idle.
  valkey    ElastiCache Serverless. Faster, and the right answer at high
            request rates, but it lives in a VPC, so the interceptor must join
            that VPC and reach every other service it calls through endpoints.
"""
import hashlib
import json
import os
import time

from ..pipeline import Call, Plugin, Serve, register
from ..state import store

_valkey = None


def _client():
    """One cluster-mode client per container. ElastiCache Serverless requires TLS."""
    global _valkey
    if _valkey is None:
        from redis.cluster import RedisCluster

        _valkey = RedisCluster(host=os.environ["VALKEY_ENDPOINT"], port=6379, ssl=True,
                               socket_timeout=2, socket_connect_timeout=2,
                               decode_responses=True)
    return _valkey


def _backend(params: dict | None = None) -> str:
    """Config wins over the deployment default, so both can be compared live."""
    return (params or {}).get("backend") or os.environ.get("CACHE_BACKEND", "dynamodb")


def _get(key: str, params=None):
    if _backend(params) == "valkey":
        raw = _client().get(f"cache#{key}")
        return json.loads(raw) if raw else None
    return store(os.environ.get("STATE_TABLE")).get_cached(key)


def _put(key: str, body: dict, ttl_s: int, params=None) -> None:
    if _backend(params) == "valkey":
        _client().setex(f"cache#{key}", ttl_s, json.dumps(body))
    else:
        store(os.environ.get("STATE_TABLE")).put_cached(key, body, ttl_s=ttl_s)


def cache_key(call: Call, per_tenant: bool = True) -> str:
    """The identity of an answer: the request as the client sent it.

    Shared with the semantic cache so that both write into the same entries --
    whichever plugin stores an answer, the other can serve it.
    """
    material = json.dumps({
        "model": call.model,
        "messages": call.body.get("messages") or call.body.get("input"),
        "temperature": call.body.get("temperature"),
        "max_tokens": call.body.get("max_tokens"),
        "tenant": call.tenant if per_tenant else "*",
    }, sort_keys=True, default=str)
    return hashlib.sha256(material.encode()).hexdigest()[:32]


@register
class Cache(Plugin):
    name = "cache"
    needs_response = True

    def _key(self, call: Call) -> str:
        return cache_key(call, self.params.get("per_tenant", True))

    def on_request(self, call: Call) -> Serve | None:
        if call.body.get("stream"):
            return None                      # a cached body is not an event stream
        key = self._key(call)
        call.attrs["cache_key"] = key
        # The response phase receives no request, so hand the key forward.
        call.attrs.setdefault("remember", {})["cache_key"] = key
        hit = _get(key, self.params)
        if hit is None:
            call.attrs["cache"] = "miss"
            return None
        call.attrs["cache"] = "hit"
        hit["x_gateway"] = {"cached": True, "tenant": call.tenant}
        return Serve(body=hit)

    def on_response(self, call: Call) -> None:
        key = call.attrs.get("cache_key") or call.attrs.get("recalled", {}).get("cache_key")
        body = call.attrs.get("replacement_body") or call.response
        if not key or not body or call.attrs.get("streamed") or not body.get("choices"):
            return
        _put(key, body, ttl_s=int(self.params.get("ttl_s", 3600)), params=self.params)
        call.attrs["cached_at"] = int(time.time())
        call.attrs["cache_backend"] = _backend(self.params)
