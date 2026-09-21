"""Scoring side of the trained router, packaged with the interceptor.

Deliberately dependency-free: the weights are JSON and the arithmetic is a dot
product over 256 floats, so the Lambda needs no numpy layer and no container
image. The only remote call is one Titan embedding per request — that hop is
the router's real latency cost, and it is measured rather than assumed.
"""

import json
import math
import os
import pathlib

import boto3

_ARTIFACT = pathlib.Path(__file__).parent / "artifacts" / "router_weights.json"
_client = None


def _bedrock():
    global _client
    if _client is None:
        _client = boto3.client("bedrock-runtime",
                               region_name=os.environ.get("AWS_REGION", "us-east-1"))
    return _client


def threshold_for(call_rate_pct: int) -> float | None:
    """Score threshold that routes roughly `call_rate_pct`% of traffic to strong.

    A fixed 0.5 is wrong here and silently so: the trained scores are bounded
    well below it by the 9.4% class imbalance, so 0.5 routes nothing at all.
    Thresholds come from the held-out score distribution instead.
    """
    spec = json.loads(_ARTIFACT.read_text())
    table = spec.get("operating_points", {}).get("thresholds_by_call_rate_pct", {})
    return table.get(str(int(call_rate_pct)))


def load():
    """Return callable(text) -> P(strong model is needed). Raises if no artifact."""
    spec = json.loads(_ARTIFACT.read_text())
    weights, bias = spec["weights"], spec["bias"]
    model_id, dims = spec["embedding_model"], spec["dims"]

    def score(text: str) -> float:
        body = json.dumps({"inputText": text[:8000], "dimensions": dims, "normalize": True})
        resp = _bedrock().invoke_model(modelId=model_id, body=body)
        vec = json.loads(resp["body"].read())["embedding"]
        z = bias + sum(v * w for v, w in zip(vec, weights))
        return 1 / (1 + math.exp(-max(min(z, 60), -60)))

    return score
