"""Importing this package registers every plugin with the pipeline."""
from . import (  # noqa: F401
    budget,
    cache,
    escalate,
    guardrail,
    max_tokens,
    metering,
    model_policy,
    router,
    tenant,
)
