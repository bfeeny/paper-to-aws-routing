"""Serve a stored answer to a question that is *equivalent*, not identical.

An exact cache only fires when two requests match byte for byte, which real
traffic rarely does: a trailing "please", a reordered clause or a different
capital letter is a miss. Semantic caching is the obvious fix -- embed the
prompt, find the nearest stored one, serve its answer if they are close enough.

The obvious fix does not work on its own. Measured on 40 prompts with Titan
embeddings, paraphrases of a question sat at a median cosine of 0.836 while
minimally edited prompts *with different answers* sat at 0.911: the dangerous
neighbors are closer than the safe ones, because changing one number barely
moves an embedding while rewording moves it a lot. No threshold separates them.

So similarity is used only to *recall* a candidate, and a cheap model decides
whether the two questions actually have the same answer. Recall is deliberately
generous -- a low threshold and a few candidates -- because a candidate that is
never recalled can never be served; precision is the verifier's job.

Recall runs on DynamoDB's own vector index, declared on the same table that
already holds the spend ledger, the correlation records and the cached answers.
That is the reason to prefer it here over a separate vector store: no second
service in the request path, no second IAM policy, nothing else to keep warm,
and the answer the index points at is one GetItem away in the same table.

One cost of sharing the table: the indexed attribute name is reserved for every
item in it. Once `embedding` is a vector attribute, no item may carry an
`embedding` of any other type -- a correlation record that stored the vector as
a JSON string under that name had its writes rejected until it was renamed.
"""
import json
import os

from ..pipeline import Call, Plugin, Serve, register
from ..state import store
from .cache import _get, _put, cache_key

_bedrock = None


def _client():
    global _bedrock
    if _bedrock is None:
        import boto3
        _bedrock = boto3.client("bedrock-runtime")
    return _bedrock


def _embed(text: str, dims: int = 256) -> list[float]:
    """Titan v2 is Matryoshka-trained, so 256 dimensions keep most of the
    ranking quality of 1024 at a quarter of the item size -- and ranking is all
    that is being asked of the embedding here."""
    r = _client().invoke_model(modelId="amazon.titan-embed-text-v2:0", body=json.dumps(
        {"inputText": text[:8000], "dimensions": dims, "normalize": True}))
    return json.loads(r["body"].read())["embedding"]


VERIFY = ("Two questions follow. Answer YES only if any correct answer to A is also a correct "
          "answer to B -- identical meaning, same facts, same numbers. Answer NO if any detail "
          "that changes the answer differs. Reply with one word: YES or NO.\n\nA: {a}\n\nB: {b}")


def _equivalent(model: str, a: str, b: str) -> bool:
    r = _client().converse(
        modelId=model,
        messages=[{"role": "user", "content": [{"text": VERIFY.format(a=a[:3000], b=b[:3000])}]}],
        inferenceConfig={"maxTokens": 5, "temperature": 0})
    text = "".join(c.get("text", "") for c in r["output"]["message"]["content"])
    return text.strip().upper().startswith("YES")


@register
class SemanticCache(Plugin):
    name = "semantic_cache"
    needs_response = True

    def _store(self):
        return store(os.environ.get("STATE_TABLE"))

    def _index(self) -> str:
        return self.params.get("index") or os.environ.get("VECTOR_INDEX", "semantic-cache")

    def on_request(self, call: Call) -> Serve | None:
        if call.body.get("stream") or call.attrs.get("cache") == "hit":
            return None                      # a stream cannot be served from a body;
        prompt = call.prompt_text()          # an exact hit has already answered
        if not prompt.strip():
            return None
        vec = _embed(prompt, int(self.params.get("dims", 256)))
        call.attrs["embedding"] = vec        # reused on the way out
        # The same key the exact cache uses, computed here too so this plugin
        # works with or without it in the chain.
        call.attrs.setdefault("cache_key", cache_key(call, self.params.get("per_tenant", True)))
        remember = call.attrs.setdefault("remember", {})
        remember.setdefault("cache_key", call.attrs["cache_key"])
        # NOT "embedding": declaring a vector index reserves that attribute
        # name across the whole table, and a correlation record storing the
        # vector as a JSON string under it is rejected by PutItem.
        remember["sem_vec"] = json.dumps([round(x, 6) for x in vec])
        remember["prompt"] = prompt[:2000]

        try:
            candidates = self._store().nearest(self._index(), call.tenant, vec,
                                               top_k=int(self.params.get("top_k", 3)))
        except Exception as exc:  # noqa: BLE001 - a broken index is a miss, not an error
            # Recorded, not swallowed: a cache that silently never hits looks
            # exactly like a cache that is working and finding nothing.
            call.attrs["semantic_cache"] = "recall_failed"
            call.attrs["semantic_recall_error"] = repr(exc)[:200]
            return None
        if not candidates:
            call.attrs["semantic_cache"] = "empty"
            call.attrs["semantic_index"] = self._index()
            return None
        call.attrs["semantic_similarity"] = round(candidates[0]["similarity"], 4)

        threshold = float(self.params.get("threshold", 0.80))
        verify = self.params.get("verify", True)
        verifier = self.params.get("verifier", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
        checked = 0
        for cand in candidates:
            if cand["similarity"] < threshold:
                break                        # sorted by similarity: the rest are worse
            body = _get(cand["cache_key"], self.params)
            if not body:
                continue                     # the answer expired; its vector outlived it
            if verify:
                checked += 1
                if not _equivalent(verifier, cand["prompt"], prompt):
                    continue
            call.attrs["semantic_cache"] = "hit"
            call.attrs["semantic_verified"] = bool(verify)
            call.attrs["semantic_verifier_calls"] = checked
            body = dict(body)
            body["x_gateway"] = {"cached": True, "semantic": True,
                                 "similarity": round(cand["similarity"], 4),
                                 "tenant": call.tenant}
            return Serve(body=body)

        call.attrs["semantic_cache"] = "rejected_by_verifier" if checked else "below_threshold"
        call.attrs["semantic_verifier_calls"] = checked
        return None

    def on_response(self, call: Call) -> None:
        recalled = call.attrs.get("recalled", {})
        key = call.attrs.get("cache_key") or recalled.get("cache_key")
        vec = call.attrs.get("embedding") or json.loads(recalled.get("sem_vec") or "null")
        prompt = call.attrs.get("prompt") or recalled.get("prompt", "")
        body = call.attrs.get("replacement_body") or call.response
        if not (key and vec and body and body.get("choices")) or call.attrs.get("streamed"):
            return
        ttl_s = int(self.params.get("ttl_s", 3600))
        _put(key, body, ttl_s=ttl_s, params=self.params)
        self._store().put_vector(key, call.tenant, vec, prompt, ttl_s=ttl_s)
        call.attrs["semantic_cache_stored"] = True
