"""REQUEST interceptor: choose the model for each inference request.

The gateway routes on the `model` field, so a router is simply a function that
rewrites that field before routing. Callers always send the virtual model ID
(`auto`); the arm configured on this stack decides what it resolves to.

Arms:
  passthrough   leave the request alone (control: measures interceptor overhead only)
  always_strong pin every request to the strong model (quality ceiling)
  always_weak   pin every request to the weak model (cost floor)
  length        input-size threshold, copied from the routing example in the
                AgentCore interceptor docs (SONNET_THRESHOLD = 2000 chars).
                A docs example, not an AWS recommendation — which is exactly
                why it is the baseline: it is what a developer would paste in.
  routellm      preference-trained router scored against ROUTER_THRESHOLD

Every decision is emitted as a CloudWatch EMF record so an experiment run can be
reconstructed from logs alone, including fallbacks.
"""

import base64
import json
import logging
import os
import time

logger = logging.getLogger()
logger.setLevel(logging.INFO)

VIRTUAL_MODEL = "auto"
PASS_THROUGH = {"interceptorOutputVersion": "1.0", "http": {}}

STRATEGY = os.environ.get("ROUTER_STRATEGY", "passthrough")
STRONG = os.environ["STRONG_MODEL"]
WEAK = os.environ["WEAK_MODEL"]
TARGET = os.environ.get("TARGET_NAME", "mantle")
LENGTH_THRESHOLD = int(float(os.environ.get("LENGTH_THRESHOLD_CHARS", "2000")))
# A raw score threshold if you truly want one, otherwise a target call rate the
# artifact resolves against its own score distribution. The old default of 0.5
# routed 0% of traffic to the strong model -- every `routellm` run before this
# was an always_weak run with an embedding call bolted on.
ROUTER_THRESHOLD = os.environ.get("ROUTER_THRESHOLD")
ROUTER_CALL_RATE = int(float(os.environ.get("ROUTER_CALL_RATE_PCT", "30")))
NAMESPACE = os.environ.get("METRIC_NAMESPACE", "routingstudy/Router")

# Loaded lazily so a missing artifact degrades visibly instead of failing the request.
_scorer = None
_scorer_error = None


def _qualify(model: str) -> str:
    """Pin the destination explicitly: unqualified IDs can collide across targets."""
    return model if "/" in model else f"{TARGET}/{model}"


def _prompt_text(payload: dict) -> str:
    """Extract user text from either the chat-completions or responses shape."""
    if isinstance(payload.get("messages"), list):
        parts = []
        for m in payload["messages"]:
            c = m.get("content")
            if isinstance(c, str):
                parts.append(c)
            elif isinstance(c, list):  # content blocks
                parts.extend(b.get("text", "") for b in c if isinstance(b, dict))
        return "\n".join(parts)
    return json.dumps(payload.get("input", ""))


def _load_scorer():
    """Load the trained router. Returns a callable(text) -> P(strong needed)."""
    global _scorer, _scorer_error
    if _scorer is not None or _scorer_error is not None:
        return _scorer
    try:
        from routellm_scorer import load  # packaged alongside this handler

        _scorer = load()
    except Exception as exc:  # noqa: BLE001 - surfaced as a metric, never raised
        _scorer_error = repr(exc)
        logger.error("router artifact unavailable, falling back to strong: %s", _scorer_error)
    return _scorer


_threshold_cache = None


def _threshold() -> float | None:
    """Resolve the score cut once: pinned value, else the calibrated call rate."""
    global _threshold_cache
    if _threshold_cache is None:
        if ROUTER_THRESHOLD is not None:
            _threshold_cache = float(ROUTER_THRESHOLD)
        else:
            from routellm_scorer import threshold_for
            _threshold_cache = threshold_for(ROUTER_CALL_RATE)
    return _threshold_cache


def decide(text: str) -> tuple[str, float | None, str]:
    """Return (model, score, decision_kind)."""
    if STRATEGY == "always_strong":
        return STRONG, None, "pinned"
    if STRATEGY == "always_weak":
        return WEAK, None, "pinned"
    if STRATEGY == "length":
        return (STRONG if len(text) >= LENGTH_THRESHOLD else WEAK), float(len(text)), "length"
    if STRATEGY == "routellm":
        scorer = _load_scorer()
        if scorer is None:
            # Fail toward quality, and make the fallback countable.
            return STRONG, None, "fallback_no_artifact"
        score = float(scorer(text))
        cut = _threshold()
        if cut is None:
            return STRONG, score, "fallback_no_threshold"
        return (STRONG if score >= cut else WEAK), score, "scored"
    return "", None, "passthrough"


def _emit(model: str, score: float | None, kind: str, elapsed_ms: float, chars: int) -> None:
    """CloudWatch EMF: metrics plus the per-request fields the analysis needs."""
    record = {
        "_aws": {
            "Timestamp": int(time.time() * 1000),
            "CloudWatchMetrics": [
                {
                    "Namespace": NAMESPACE,
                    "Dimensions": [["strategy"], ["strategy", "model"]],
                    "Metrics": [
                        {"Name": "RouterLatencyMs", "Unit": "Milliseconds"},
                        {"Name": "Decisions", "Unit": "Count"},
                        {"Name": "PromptChars", "Unit": "Count"},
                    ],
                }
            ],
        },
        "strategy": STRATEGY,
        "model": model or "unchanged",
        "decision": kind,
        "score": score,
        "RouterLatencyMs": elapsed_ms,
        "Decisions": 1,
        "PromptChars": chars,
    }
    print(json.dumps(record))


def lambda_handler(event, context):
    started = time.perf_counter()
    http = event.get("http", {})
    encoded = http.get("gatewayRequest", {}).get("body")
    if not encoded:
        return PASS_THROUGH

    try:
        payload = json.loads(base64.b64decode(encoded))
    except (ValueError, TypeError) as exc:
        logger.warning("unparseable body, passing through: %s", exc)
        return PASS_THROUGH

    # Only the virtual model is ours to resolve; anything explicit routes normally,
    # which keeps a stack usable for ad-hoc calls during a run.
    if not isinstance(payload, dict) or payload.get("model") != VIRTUAL_MODEL:
        return PASS_THROUGH

    text = _prompt_text(payload)
    model, score, kind = decide(text)
    elapsed_ms = (time.perf_counter() - started) * 1000
    _emit(model, score, kind, elapsed_ms, len(text))

    if not model:  # passthrough arm: leave the alias alone
        return PASS_THROUGH

    payload["model"] = _qualify(model)
    return {
        "interceptorOutputVersion": "1.0",
        "http": {
            "transformedGatewayRequest": {
                "body": base64.b64encode(json.dumps(payload).encode()).decode()
            }
        },
    }
