"""Resolve who is calling, so later plugins can apply per-tenant policy.

The demo stack reads a request header. In production the tenant should come
from the authorizer -- a claim in a validated JWT -- never from a header the
caller can set; this plugin is the single place that decision lives.
"""
from ..pipeline import Call, Plugin, Reject, register


@register
class Tenant(Plugin):
    name = "tenant"

    def on_request(self, call: Call) -> Reject | None:
        header = self.params.get("header", "x-tenant-id").lower()
        value = (call.headers.get(header) or "").strip()
        known = self.params.get("known")
        if not value:
            if self.params.get("require"):
                return Reject(401, "tenant_required", f"missing {header} header")
            value = self.params.get("default", "anonymous")
        elif known and value not in known:
            return Reject(403, "unknown_tenant", f"tenant {value!r} is not provisioned")
        call.tenant = value
        return None

    on_response = on_request  # the response phase needs the same identity
