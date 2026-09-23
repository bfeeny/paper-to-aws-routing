"""Derive the tenant from the caller's token instead of from a header.

`x-tenant-id: acme` is a request to be treated as Acme. Any client can send it,
including Acme's competitor, so every per-tenant control built on it -- the
budget, the rate limit, the cache partition, the vector index partition -- is
enforcing a preference rather than a boundary. The demo pipeline shipped that
way and said so; this plugin is the fix.

When the gateway's authorizer type is CUSTOM_JWT, a request cannot reach the
interceptor at all unless the gateway has already validated the token against
the issuer's JWKS and checked the audience and client. The tenant is then a
claim in a token that a third party signed and the platform verified, which is
a different kind of fact from a header.

That raises a real question: should the interceptor verify the signature again?

  verify: true   (default) Fetch the issuer's JWKS, cache it per container, and
                 check signature, expiry, issuer and audience independently.
                 Costs one HTTPS round trip on a cold cache and a few hundred
                 microseconds thereafter. This is what you want if the tenant
                 decides who gets billed, because it does not depend on the
                 gateway being configured the way you remember.
  verify: false  Decode the claims and trust the gateway's validation. Cheaper,
                 and correct exactly as long as the authorizer stays CUSTOM_JWT.
                 The failure mode is silent: switch the gateway back to
                 AWS_IAM for a test and this plugin will happily read the
                 tenant out of an unsigned token anyone can mint.

The default is to verify, because the cost is small and measurable and the
failure mode of the alternative is not.
"""
import json
import os
import time
import urllib.request

from ..pipeline import Call, Plugin, Reject, register

_jwks: dict = {"at": 0.0, "url": None, "keys": {}}


def _bearer(headers: dict) -> str | None:
    raw = headers.get("authorization") or headers.get("Authorization") or ""
    parts = raw.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1]
    return None


def _jwks_for(issuer: str, ttl: float = 3600.0) -> dict:
    """Public keys for an issuer, cached per container and keyed by `kid`.

    Key rotation is why this expires: a cache that never refreshes turns the
    issuer's routine rotation into a total outage some weeks later.
    """
    url = issuer.rstrip("/") + "/.well-known/jwks.json"
    if _jwks["url"] == url and time.time() - _jwks["at"] < ttl and _jwks["keys"]:
        return _jwks["keys"]
    with urllib.request.urlopen(url, timeout=3) as r:
        doc = json.load(r)
    _jwks.update(at=time.time(), url=url,
                 keys={k["kid"]: k for k in doc.get("keys", []) if "kid" in k})
    return _jwks["keys"]


def _decode_unverified(token: str) -> dict:
    import jwt
    return jwt.decode(token, options={"verify_signature": False})


def _decode_verified(token: str, issuer: str, audience: list[str] | None) -> dict:
    import jwt
    from jwt.algorithms import RSAAlgorithm

    kid = jwt.get_unverified_header(token).get("kid")
    keys = _jwks_for(issuer)
    if kid not in keys:
        # A key the cache has not seen is the expected shape of a rotation.
        _jwks["at"] = 0.0
        keys = _jwks_for(issuer)
    if kid not in keys:
        raise ValueError(f"signing key {kid!r} is not published by {issuer}")
    key = RSAAlgorithm.from_jwk(json.dumps(keys[kid]))
    return jwt.decode(token, key=key, algorithms=["RS256"], issuer=issuer,
                      audience=audience,
                      options={"verify_aud": bool(audience), "require": ["exp", "iss"]})


@register
class JwtTenant(Plugin):
    name = "jwt_tenant"

    def _tenant_from(self, claims: dict) -> str | None:
        # Cognito's machine-to-machine tokens carry no custom claims, so the
        # conventional place to put tenancy is a resource-server scope:
        # `gw/tenant-acme` names both the resource and the tenant. When a
        # scope prefix is configured it wins, because it is the more specific
        # statement: the client id says who is calling, the scope says what
        # they were issued the right to act as.
        value = None
        prefix = self.params.get("scope_prefix")
        if prefix:
            for s in (claims.get("scope") or "").split():
                name = s.rsplit("/", 1)[-1]
                if name.startswith(prefix):
                    value = name[len(prefix):]
                    break
        if value is None:
            value = claims.get(self.params.get("claim", "client_id"))
        if value is None:
            return None
        mapping = self.params.get("map")
        return str(mapping.get(str(value), value)) if mapping else str(value)

    def on_request(self, call: Call) -> Reject | None:
        token = _bearer(call.headers)
        if not token:
            if self.params.get("require", True):
                return Reject(401, "no_token", "a bearer token is required")
            call.attrs["jwt"] = "absent"
            return None

        issuer = self.params.get("issuer") or os.environ.get("JWT_ISSUER", "")
        audience = self.params.get("audience")
        try:
            if self.params.get("verify", True):
                if not issuer:
                    return Reject(500, "jwt_misconfigured",
                                  "verification is on but no issuer is configured")
                claims = _decode_verified(token, issuer, audience)
                call.attrs["jwt"] = "verified"
            else:
                claims = _decode_unverified(token)
                call.attrs["jwt"] = "unverified"
        except Exception as exc:  # noqa: BLE001 - any failure here is a refusal
            return Reject(401, "invalid_token", f"token rejected: {type(exc).__name__}")

        tenant = self._tenant_from(claims)
        if not tenant:
            return Reject(403, "no_tenant_claim",
                          f"token carries no {self.params.get('claim', 'client_id')!r} claim")
        known = self.params.get("known")
        if known and tenant not in known:
            return Reject(403, "unknown_tenant", f"tenant {tenant!r} is not provisioned")
        call.tenant = tenant
        # Handy in the logs, and safe: an identifier, never the token.
        call.attrs["jwt_sub"] = str(claims.get("sub") or claims.get("client_id") or "")[:64]
        return None
