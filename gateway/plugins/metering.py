"""Attribute tokens and dollars to the tenant and model that incurred them.

Reads the provider's own usage block from the response, so the numbers are the
ones Bedrock bills, not an estimate from the prompt. Emits them as CloudWatch
metrics dimensioned by tenant and model, which is enough to build a per-tenant
chargeback view without a separate ledger.

High-cardinality tenant IDs make for expensive custom metrics; beyond a few
hundred tenants, keep tenant in the log record and aggregate with Logs
Insights instead of as a metric dimension.
"""
import json
import os
import time

from ..pipeline import Call, Plugin, register
from ..prices import bare, cost_usd


@register
class Metering(Plugin):
    name = "metering"

    def on_response(self, call: Call) -> None:
        usage = (call.response or {}).get("usage") or {}
        tin = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        tout = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        cost = cost_usd(call.model, tin, tout)
        call.attrs.update(input_tokens=tin, output_tokens=tout)
        if cost is not None:
            call.attrs["cost_usd"] = round(cost, 8)
        ns = self.params.get("namespace") or os.environ.get("METRIC_NAMESPACE", "gateway/Pipeline")
        metrics = [{"Name": "InputTokens", "Unit": "Count"},
                   {"Name": "OutputTokens", "Unit": "Count"}]
        if cost is not None:
            metrics.append({"Name": "CostUsd", "Unit": "None"})
        print(json.dumps({
            "_aws": {"Timestamp": int(time.time() * 1000),
                     "CloudWatchMetrics": [{"Namespace": ns,
                                            "Dimensions": [["model"], ["tenant"], ["tenant", "model"]],
                                            "Metrics": metrics}]},
            "tenant": call.tenant, "model": bare(call.model),
            "InputTokens": tin, "OutputTokens": tout,
            **({"CostUsd": cost} if cost is not None else {}),
        }))
