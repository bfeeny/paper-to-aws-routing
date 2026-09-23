"""Importing this package registers every plugin with the pipeline."""
from . import (  # noqa: F401
    budget,
    cache,
    escalate,
    guardrail,
    jwt_tenant,
    max_tokens,
    metering,
    model_policy,
    pii,
    rate_limit,
    router,
    semantic_cache,
    tenant,
)
