"""Screen the prompt with an Amazon Bedrock guardrail before any model sees it.

Uses the standalone ApplyGuardrail API, so the same guardrail applies whichever
model the router later chooses, and a blocked prompt costs a guardrail check
rather than a model call. Only the user text is sent; very long prompts are
truncated to `max_chars` to bound the per-request cost.
"""
from ..pipeline import Call, Plugin, Reject, register

_client = None


def _bedrock():
    global _client
    if _client is None:
        import boto3
        _client = boto3.client("bedrock-runtime")
    return _client


@register
class Guardrail(Plugin):
    name = "guardrail"

    def on_request(self, call: Call) -> Reject | None:
        text = call.prompt_text()[: int(self.params.get("max_chars", 8000))]
        if not text.strip():
            return None
        r = _bedrock().apply_guardrail(
            guardrailIdentifier=self.params["guardrail_id"],
            guardrailVersion=str(self.params.get("version", "DRAFT")),
            source="INPUT",
            content=[{"text": {"text": text}}],
        )
        call.attrs["guardrail_action"] = r.get("action")
        if r.get("action") == "GUARDRAIL_INTERVENED":
            msg = (r.get("outputs") or [{}])[0].get("text") or "blocked by policy"
            return Reject(400, "guardrail_intervened", msg)
        return None
