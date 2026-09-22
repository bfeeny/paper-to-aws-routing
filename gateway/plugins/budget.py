"""Per-tenant daily spend limits, enforced before the call and settled after it.

REQUEST phase: read today's spend for the tenant and refuse the call once the
limit is reached. With `reserve: true` it also refuses a call whose worst case
(its full max_tokens at the model's output price) would cross the limit -- a
stricter policy that never overshoots, at the price of rejecting a few calls
that would have fit.

RESPONSE phase: add the actual cost, computed from the provider's reported
usage, with an atomic DynamoDB ADD so concurrent calls cannot lose updates.

Between the check and the settlement, concurrent requests can each pass the
check and together overshoot by up to (concurrency x one call). That is the
usual trade for not serializing traffic through a lock; the alternative is
reservations, above.
"""
import datetime as dt
from decimal import Decimal

from ..pipeline import Call, Plugin, Reject, register
from ..prices import cost_usd, output_price_per_1k

_table = None


def _ddb(name: str):
    global _table
    if _table is None:
        import boto3
        _table = boto3.resource("dynamodb").Table(name)
    return _table


def _key(tenant: str) -> str:
    return f"{tenant}#{dt.datetime.now(dt.timezone.utc):%Y-%m-%d}"


@register
class Budget(Plugin):
    name = "budget"

    def _limit(self, tenant: str) -> float:
        limits = self.params.get("daily_usd", {})
        return float(limits.get(tenant, limits.get("*", 10.0)))

    def on_request(self, call: Call) -> Reject | None:
        item = _ddb(self.params["table"]).get_item(Key={"pk": _key(call.tenant)}).get("Item")
        spent = float(item["spend_usd"]) if item else 0.0
        limit = self._limit(call.tenant)
        call.attrs["budget_spent_usd"] = round(spent, 6)
        if spent >= limit:
            return Reject(429, "budget_exceeded",
                          f"daily budget of ${limit:.2f} reached for {call.tenant!r}")
        if self.params.get("reserve"):
            price = output_price_per_1k(call.model)
            worst = (price or 0) * int(call.body.get("max_tokens") or 0) / 1000
            if spent + worst > limit:
                return Reject(429, "budget_would_exceed",
                              f"this request could cost ${worst:.4f}; "
                              f"${limit - spent:.4f} of today's budget remains")
        return None

    def on_response(self, call: Call) -> None:
        cost = call.attrs.get("cost_usd")
        if cost is None:  # metering has not run or the model is unpriced
            usage = (call.response or {}).get("usage") or {}
            cost = cost_usd(call.model, int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0),
                            int(usage.get("completion_tokens") or usage.get("output_tokens") or 0))
        if not cost:
            return
        _ddb(self.params["table"]).update_item(
            Key={"pk": _key(call.tenant)},
            UpdateExpression="ADD spend_usd :c SET #t = :ttl",
            ExpressionAttributeNames={"#t": "expires_at"},
            ExpressionAttributeValues={
                ":c": Decimal(str(round(cost, 8))),
                # keep a week of history, then let DynamoDB TTL remove it
                ":ttl": int((dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=7)).timestamp()),
            },
        )
