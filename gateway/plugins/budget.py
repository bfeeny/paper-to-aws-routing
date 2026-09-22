"""Per-tenant daily spend limits, enforced before the call and settled after it.

REQUEST phase: read today's spend and refuse the call once the limit is
reached. With `reserve: true` it also refuses a call whose worst case (its full
max_tokens at the model's output price) would cross the limit -- a stricter
policy that never overshoots, at the price of rejecting a few calls that would
have fit. Place it after the router, so the price is the resolved model's.

RESPONSE phase: charge the actual cost from the provider's reported usage.
Settlement is idempotent per REQUEST_ID (see state.py), because the gateway may
retry an interceptor and a retried charge must not bill twice.

Between check and settlement, concurrent calls can each pass the check and
together overshoot by up to (concurrency x one call). That is the usual price
of not serializing traffic through a lock; reservations are the alternative.
"""
import os

from ..pipeline import Call, Plugin, Reject, register
from ..prices import output_price_per_1k
from ..state import store


@register
class Budget(Plugin):
    name = "budget"
    needs_response = True

    def _limit(self, tenant: str) -> float:
        limits = self.params.get("daily_usd", {})
        return float(limits.get(tenant, limits.get("*", 10.0)))

    def on_request(self, call: Call) -> Reject | None:
        spent = store(self.params.get("table") or os.environ.get("STATE_TABLE")).spend_today(call.tenant)
        limit = self._limit(call.tenant)
        call.attrs["budget_spent_usd"] = round(spent, 6)
        if spent >= limit:
            return Reject(429, "budget_exceeded",
                          f"daily budget of ${limit:.2f} reached for {call.tenant!r}")
        if self.params.get("reserve"):
            price = output_price_per_1k(call.model) or 0
            worst = price * int(call.body.get("max_tokens") or 0) / 1000
            if spent + worst > limit:
                return Reject(429, "budget_would_exceed",
                              f"this request could cost ${worst:.4f}; "
                              f"${limit - spent:.4f} of today's budget remains")
        return None

    def on_response(self, call: Call) -> None:
        cost = call.attrs.get("cost_usd")
        if not cost or not call.request_id:
            return
        charged = store(self.params.get("table") or os.environ.get("STATE_TABLE")).settle(call.request_id, call.tenant, cost)
        call.attrs["budget_settled"] = charged
