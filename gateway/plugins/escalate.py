"""Retry a weak answer on a strong model, and return that answer instead.

A RESPONSE interceptor can replace the body the client receives, and nothing
stops it calling a model itself. Together those allow escalation: let the cheap
model answer, look at what came back, and if it looks like a failure, ask the
expensive model and return its answer. The client sees one call.

This is the deployable form of a result from the companion routing study: a
prompt-only router barely beat prompt length at predicting which requests need
the strong model, while the cheap model's own output -- how long it ran, and
whether it stopped because it hit the token ceiling -- predicted it far better.
Deciding after the answer exists is simply an easier problem than guessing
before it.

The costs are real and worth stating: an escalated request pays for both calls
and waits for them in series. Escalation is for traffic where a wrong answer
costs more than a slow one.

Requires the request phase to remember the prompt, since the response phase
never receives it.
"""
import json
import os
import urllib.error
import urllib.request

from ..pipeline import Call, Plugin, register
from ..prices import bare
from ..state import store

MAX_REMEMBERED = 12000


def _mantle(model: str, messages: list, max_tokens: int, region: str) -> dict | None:
    import boto3
    from botocore.auth import SigV4Auth
    from botocore.awsrequest import AWSRequest

    url = f"https://bedrock-mantle.{region}.api.aws/v1/chat/completions"
    payload = json.dumps({"model": bare(model), "messages": messages,
                          "max_tokens": max_tokens, "temperature": 0}).encode()
    req = AWSRequest(method="POST", url=url, data=payload,
                     headers={"Content-Type": "application/json"})
    creds = boto3.Session().get_credentials().get_frozen_credentials()
    SigV4Auth(creds, "bedrock-mantle", region).add_auth(req)
    try:
        with urllib.request.urlopen(
            urllib.request.Request(url, data=payload, headers=dict(req.headers)), timeout=60
        ) as r:
            return json.load(r)
    except urllib.error.HTTPError:
        return None


@register
class Escalate(Plugin):
    name = "escalate"
    needs_response = True

    def on_request(self, call: Call):
        # The response phase gets no request, so keep what a second call needs.
        messages = call.body.get("messages")
        if messages:
            blob = json.dumps(messages)
            if len(blob) <= MAX_REMEMBERED:
                call.attrs.setdefault("remember", {})["messages"] = blob
                call.attrs["remember"]["max_tokens"] = str(call.body.get("max_tokens") or 1024)
        return None

    def _reason(self, body: dict) -> str | None:
        choice = (body.get("choices") or [{}])[0]
        usage = body.get("usage") or {}
        out = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        if choice.get("finish_reason") == "length":
            return "truncated"
        if out >= int(self.params.get("verbose_tokens", 400)):
            return "verbose"
        text = (choice.get("message") or {}).get("content") or ""
        if len(text.strip()) < int(self.params.get("min_chars", 0)):
            return "too_short"
        return None

    def on_response(self, call: Call) -> None:
        if call.attrs.get("streamed") or not call.response:
            return                      # cannot rewrite an event stream
        strong = self.params["strong"]
        if bare(call.model) == bare(strong):
            return                      # already the strong model
        reason = self._reason(call.response)
        call.attrs["escalation_reason"] = reason or "none"
        if not reason:
            return
        seen = store(self.params.get("table") or os.environ.get("STATE_TABLE")).recall(call.request_id)
        if not seen.get("messages"):
            call.attrs["escalation_reason"] = "prompt_not_remembered"
            return
        better = _mantle(strong, json.loads(seen["messages"]),
                         int(seen.get("max_tokens") or 1024),
                         os.environ.get("AWS_REGION", "us-east-1"))
        if not better:
            call.attrs["escalation_reason"] = f"{reason}_but_strong_call_failed"
            return
        usage = call.response.get("usage") or {}
        call.attrs["extra_usage"] = {
            "model": call.model,
            "in": int(usage.get("prompt_tokens") or 0),
            "out": int(usage.get("completion_tokens") or 0),
        }
        call.attrs["escalated_from"] = bare(call.model)
        call.attrs["escalated_to"] = bare(strong)
        call.attrs["replacement_body"] = better
        call.response = better
        call.body["model"] = strong      # price the answer that was returned
