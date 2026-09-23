"""Serve a stored answer to a question that is *equivalent*, not identical.

An exact cache only fires when two requests match byte for byte, which real
traffic rarely does: a trailing "please", a reordered clause or a different
capital letter is a miss. Semantic caching is the obvious fix -- embed the
prompt, find the nearest stored one, serve its answer if they are close enough.

The obvious fix does not work on its own. Measured on 40 prompts with Titan
embeddings, paraphrases of a question sat at a median cosine of 0.836 while
minimally edited prompts *with different answers* sat at 0.911: the dangerous
neighbours are closer than the safe ones, because changing one number barely
moves an embedding while rewording moves it a lot. No threshold separates them.

So similarity is used only to *recall* a candidate, and a cheap model decides
whether the two questions actually have the same answer. That pairing measured
85% of paraphrases served with 2% false hits, against a verifier cost of about
680 ms and a few hundred tokens -- which is why `verify` is on by default and
why this plugin is worth enabling only when the call it avoids is much more
expensive than the check.

Recall uses Amazon S3 Vectors, which is serverless and reachable from a VPC
through its own endpoint. Answers stay in the response cache's store; the
vector index holds only the embedding and the key.
"""
import json
import os
import urllib.parse
import urllib.request

from ..pipeline import Call, Plugin, Serve, register
from .cache import _get, _put

_bedrock = None


def _client():
    global _bedrock
    if _bedrock is None:
        import boto3
        _bedrock = boto3.client("bedrock-runtime")
    return _bedrock


def _s3vectors():
    import boto3
    return boto3.client("s3vectors")


def _embed(text: str, dims: int = 256) -> list[float]:
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

    def on_request(self, call: Call) -> Serve | None:
        if call.body.get("stream") or call.attrs.get("cache") == "hit":
            return None
        prompt = call.prompt_text()
        if not prompt.strip():
            return None
        vec = _embed(prompt)
        call.attrs["embedding"] = vec          # reused on the way out
        remember = call.attrs.setdefault("remember", {})
        remember["embedding"] = json.dumps(vec)
        remember["prompt"] = prompt[:2000]
        bucket, index = os.environ["VECTOR_BUCKET"], os.environ["VECTOR_INDEX"]
        try:
            r = _s3vectors().query_vectors(
                vectorBucketName=bucket, indexName=index, topK=1,
                queryVector={"float32": vec}, returnMetadata=True, returnDistance=True,
                filter={"tenant": call.tenant})
        except Exception:  # noqa: BLE001 - an empty index is a miss, not an error
            return None
        matches = r.get("vectors") or []
        if not matches:
            return None
        best = matches[0]
        similarity = 1.0 - float(best.get("distance", 1.0))
        call.attrs["semantic_similarity"] = round(similarity, 4)
        if similarity < float(self.params.get("threshold", 0.80)):
            call.attrs["semantic_cache"] = "below_threshold"
            return None
        meta = best.get("metadata") or {}
        body = _get(meta.get("key", ""), self.params)
        if not body:
            call.attrs["semantic_cache"] = "answer_expired"
            return None
        if self.params.get("verify", True) and not _equivalent(
                self.params.get("verifier", "us.anthropic.claude-haiku-4-5-20251001-v1:0"),
                meta.get("prompt", ""), prompt):
            call.attrs["semantic_cache"] = "rejected_by_verifier"
            return None
        call.attrs["semantic_cache"] = "hit"
        body = dict(body)
        body["x_gateway"] = {"cached": True, "semantic": True,
                             "similarity": call.attrs["semantic_similarity"],
                             "tenant": call.tenant}
        return Serve(body=body)

    def on_response(self, call: Call) -> None:
        key = call.attrs.get("cache_key") or call.attrs.get("recalled", {}).get("cache_key")
        vec = call.attrs.get("embedding") or json.loads(
            call.attrs.get("recalled", {}).get("embedding") or "null")
        prompt = call.attrs.get("recalled", {}).get("prompt", "")
        body = call.attrs.get("replacement_body") or call.response
        if not (key and vec and body and body.get("choices")) or call.attrs.get("streamed"):
            return
        _put(key, body, ttl_s=int(self.params.get("ttl_s", 3600)), params=self.params)
        _s3vectors().put_vectors(
            vectorBucketName=os.environ["VECTOR_BUCKET"], indexName=os.environ["VECTOR_INDEX"],
            vectors=[{"key": key, "data": {"float32": vec},
                      "metadata": {"key": key, "tenant": call.tenant, "prompt": prompt[:2000]}}])
        call.attrs["semantic_cache_stored"] = True
