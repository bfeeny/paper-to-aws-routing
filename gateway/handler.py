"""AgentCore Gateway interceptor entry point: one Lambda, both phases.

A gateway has at most one REQUEST and one RESPONSE interceptor and no native
chaining, so the plugin chain runs in-process here. This module only
translates between the gateway's interceptor contract and the pipeline:

  REQUEST   input  http.gatewayRequest {path, headers?, body(base64)}
            output http.transformedGatewayRequest {body}         continue
                   http.transformedGatewayResponse {statusCode..} short-circuit:
                   the target is not called and no RESPONSE interceptor runs
  RESPONSE  input  http.gatewayResponse {statusCode, body(base64)}, and
                   gatewayRequest null -- the request is not repeated here
            output http.transformedGatewayResponse {headers}      annotate

The gateway's REQUEST_ID arrives in the invocation's client context and is the
same for both phases; it is how the response phase recovers what the request
phase decided (see state.py).
"""
import base64
import json
import logging
import os
import time

from . import plugins  # noqa: F401  (registers every plugin)
from .pipeline import Call, emit, load_pipeline
from .state import store

log = logging.getLogger()
log.setLevel(logging.INFO)

NAMESPACE = os.environ.get("METRIC_NAMESPACE", "gateway/Pipeline")
TABLE = os.environ.get("STATE_TABLE")
PASS = {"interceptorOutputVersion": "1.0", "http": {}}
# Where to tell the caller what a call cost. The gateway drops headers set by a
# RESPONSE interceptor on inference responses (observed, not documented), so
# "body" adds a top-level field that OpenAI-compatible clients ignore.
ANNOTATE = os.environ.get("RESPONSE_ANNOTATE", "body")

# Open the state store before any plugin runs. Opening it lazily meant the
# first request in each new container reached the budget plugin with no store,
# raised, and -- the pipeline being fail-open -- skipped the budget check.
if TABLE:
    store(TABLE)


def _b64json(s):
    return json.loads(base64.b64decode(s)) if s else None


def _enc(obj) -> str:
    return base64.b64encode(json.dumps(obj).encode()).decode()


def _request_id(context) -> str:
    cc = getattr(context, "client_context", None)
    custom = getattr(cc, "custom", None) or {}
    return str(custom.get("REQUEST_ID", "")) or getattr(context, "aws_request_id", "")


def lambda_handler(event, context):
    http = event.get("http") or {}
    rid = _request_id(context)
    try:
        if http.get("gatewayResponse") is not None:
            return _on_response(http["gatewayResponse"], rid)
        return _on_request(http.get("gatewayRequest") or {}, rid)
    except Exception:  # noqa: BLE001 - never turn a pipeline bug into a failed call
        log.exception("interceptor failure; passing through")
        return PASS


def _on_request(req: dict, rid: str):
    body = _b64json(req.get("body"))
    if not isinstance(body, dict):
        return PASS
    headers = {k.lower(): v for k, v in (req.get("headers") or {}).items()}
    call = Call(body=body, headers=headers, path=req.get("path", ""), request_id=rid)
    pipeline = load_pipeline()
    reject, trace = pipeline.run_request(call)
    emit(trace, call, NAMESPACE, "rejected" if reject else "forwarded")

    if reject:
        return {"interceptorOutputVersion": "1.0", "http": {"transformedGatewayResponse": {
            "statusCode": reject.status,
            "contentType": "application/json",
            "headers": {"x-gateway-rejected-by": trace.rejected_by or "pipeline"},
            "body": _enc(reject.as_body()),
        }}}

    # Only pay for a correlation write when some plugin will need it on the way out.
    if TABLE and any(p.needs_response for p in pipeline.plugins):
        store(TABLE).remember(rid, {"tenant": call.tenant, "model": call.model,
                                    "t0": f"{time.time():.3f}"})
    return {"interceptorOutputVersion": "1.0",
            "http": {"transformedGatewayRequest": {"body": _enc(call.body)}}}


def _on_response(resp: dict, rid: str):
    pipeline = load_pipeline()
    if not any(p.needs_response for p in pipeline.plugins):
        return PASS
    seen = store(TABLE).recall(rid) if TABLE else {}
    if not seen:
        # Nothing was forwarded under this REQUEST_ID -- typically a request this
        # pipeline rejected. The gateway still invokes the RESPONSE interceptor
        # for it (observed, contrary to the documentation); there is nothing to
        # meter or settle.
        return PASS
    body = _b64json(resp.get("body")) or {}
    call = Call(body={"model": seen.get("model") or body.get("model", "")},
                tenant=seen.get("tenant", "anonymous"), request_id=rid,
                response=body if isinstance(body, dict) else {})
    trace = pipeline.run_response(call)
    emit(trace, call, NAMESPACE, f"status_{resp.get('statusCode')}")

    if "cost_usd" not in call.attrs or ANNOTATE == "none":
        return PASS
    note = {"cost_usd": call.attrs["cost_usd"], "tenant": call.tenant}
    if ANNOTATE == "headers":
        return {"interceptorOutputVersion": "1.0", "http": {"transformedGatewayResponse": {
            "headers": {"x-gateway-cost-usd": f"{note['cost_usd']:.8f}"}}}}
    if isinstance(body, dict) and body:
        body["x_gateway"] = note
        return {"interceptorOutputVersion": "1.0", "http": {"transformedGatewayResponse": {
            "body": _enc(body)}}}
    return PASS
