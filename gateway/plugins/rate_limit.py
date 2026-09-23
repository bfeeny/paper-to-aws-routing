"""Per-tenant rate limits on requests *and* on tokens.

Counting requests is the wrong unit for an LLM gateway. One request can be
eighty tokens or eighty thousand, so a limit of "100 requests per minute" is a
limit on a quantity nobody is billed for and no quota is denominated in. The
thing that actually runs out -- the model's tokens-per-minute quota, and the
bill -- is measured in tokens, and a request-rate limiter cannot see it.

So this plugin enforces two windows at once:

  requests_per_min   cheap, catches runaway loops and retry storms
  tokens_per_min     the one that maps to the quota you will actually exhaust

Tokens are charged the way Bedrock charges quota: the full `max_tokens` is
*reserved* when the request starts, and the reservation is reconciled against
real usage on the way out. Reserving is what makes the limit truthful under
concurrency -- ten simultaneous requests each asking for 4,000 tokens have
already committed 40,000 tokens of quota before any of them answers, and a
limiter that waits for actual usage will let all ten through.

The window is fixed, not sliding: one DynamoDB item per tenant per minute,
incremented with ADD and expired by TTL. A fixed window admits up to twice the
limit across a boundary, which is the usual trade for one atomic write per
call and no state to sweep. A sliding window costs a read of the previous
window per call; `smooth: true` pays that to weight the previous window and
remove the boundary burst.

Native alternatives, and why this exists anyway: AWS WAF rate-based rules
attach to the gateway and count *requests* per five-minute window, keyed on IP
or a header. They stop a flood before it reaches any of your code, which is the
right place for that job, and they cannot express "this tenant may spend 50,000
tokens a minute." Use both: WAF for volumetric abuse, this for quota.
"""
import math
import os
import time

from ..pipeline import Call, Plugin, Reject, register
from ..state import store


def _window(now: float, seconds: int) -> int:
    return int(now // seconds)


@register
class RateLimit(Plugin):
    name = "rate_limit"
    needs_response = True

    def _store(self):
        return store(self.params.get("table") or os.environ.get("STATE_TABLE"))

    def _limit(self, key: str, tenant: str, default: float) -> float:
        limits = self.params.get(key) or {}
        return float(limits.get(tenant, limits.get("*", default)))

    def on_request(self, call: Call) -> Reject | None:
        window_s = int(self.params.get("window_s", 60))
        now = time.time()
        w = _window(now, window_s)
        req_limit = self._limit("requests_per_min", call.tenant, math.inf)
        tok_limit = self._limit("tokens_per_min", call.tenant, math.inf)
        if req_limit == math.inf and tok_limit == math.inf:
            return None

        st = self._store()
        # Reserve the request's worst case before anyone else can. The counters
        # come back already including this call, so the check is on the value
        # after the increment -- no read-then-write race.
        reserve = int(call.body.get("max_tokens") or self.params.get("assume_tokens", 1024))
        counts = st.bump_window(call.tenant, w, window_s, requests=1, tokens=reserve,
                               smooth=bool(self.params.get("smooth")))
        call.attrs["rl_requests"] = counts["requests"]
        call.attrs["rl_tokens_reserved"] = counts["tokens"]   # window total after this call
        call.attrs["rl_tokens_reserved_by_me"] = reserve      # what this call must release
        call.attrs.setdefault("remember", {})["rl_reserved"] = str(reserve)
        call.attrs["remember"]["rl_window"] = str(w)

        retry_after = int((w + 1) * window_s - now) + 1
        if counts["requests"] > req_limit:
            return Reject(429, "rate_limit_requests",
                          f"{call.tenant!r} exceeded {req_limit:.0f} requests per "
                          f"{window_s}s; retry in {retry_after}s")
        if counts["tokens"] > tok_limit:
            # The reservation is what crossed the line, not measured usage; say so,
            # because the fix is often "ask for fewer max_tokens", not "send less".
            return Reject(429, "rate_limit_tokens",
                          f"{call.tenant!r} would exceed {tok_limit:.0f} tokens per "
                          f"{window_s}s (this request reserved {reserve}); "
                          f"retry in {retry_after}s")
        return None

    def on_abort(self, call: Call, verdict) -> None:
        """Release the whole reservation: this call never reaches a model.

        A cache hit is the case that makes this necessary. The answer is served
        from the interceptor, no tokens are spent, and without a release the
        tenant is charged quota for a call that never happened -- which walks
        the counter up by one full reservation per cached request until the
        tenant is locked out by traffic that cost nothing.

        The request counter is not released: the request was real, it just did
        not reach a model.
        """
        reserved = int(call.attrs.get("rl_tokens_reserved_by_me") or 0)
        window = call.attrs.get("remember", {}).get("rl_window")
        if not reserved or window is None:
            return
        self._store().bump_window(call.tenant, int(window),
                                  int(self.params.get("window_s", 60)),
                                  requests=0, tokens=-reserved)
        call.attrs["rl_released"] = reserved

    def on_response(self, call: Call) -> None:
        """Give back the difference between what was reserved and what was used."""
        recalled = call.attrs.get("recalled", {})
        # Never fall back to rl_tokens_reserved: that is the whole window's
        # total, and giving it back would zero out every other caller's usage.
        reserved = int(recalled.get("rl_reserved")
                       or call.attrs.get("rl_tokens_reserved_by_me") or 0)
        window = recalled.get("rl_window")
        used = int(call.attrs.get("input_tokens") or 0) + int(call.attrs.get("output_tokens") or 0)
        if not reserved or window is None:
            return
        delta = used - reserved
        if delta:
            self._store().bump_window(call.tenant, int(window),
                                      int(self.params.get("window_s", 60)),
                                      requests=0, tokens=delta)
        call.attrs["rl_reserved_vs_used"] = f"{reserved}/{used}"
