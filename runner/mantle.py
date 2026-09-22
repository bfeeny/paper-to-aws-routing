"""Minimal signed client for the Bedrock OpenAI-compatible (mantle) endpoint.

Used by the judge and by price lookups — anything that must reach a model
*directly*, bypassing the gateway under test. Handles the two API shapes the
endpoint exposes: OpenAI chat-completions for most models, and the Anthropic
messages API for Claude.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

SERVICE = "bedrock-mantle"


class Mantle:
    def __init__(self, profile: str = "personal", region: str = "us-east-1"):
        self.region = region
        self.base = f"https://bedrock-mantle.{region}.api.aws"
        self._creds = boto3.Session(profile_name=profile, region_name=region) \
            .get_credentials().get_frozen_credentials()

    def _post(self, path: str, payload: dict, timeout: int = 180):
        url = self.base + path
        body = json.dumps(payload).encode()
        req = AWSRequest(method="POST", url=url, data=body,
                         headers={"Content-Type": "application/json"})
        SigV4Auth(self._creds, SERVICE, self.region).add_auth(req)
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, data=body, headers=dict(req.headers)),
                timeout=timeout,
            ) as r:
                return r.status, json.load(r), (time.perf_counter() - started) * 1000
        except urllib.error.HTTPError as e:
            try:
                detail = json.loads(e.read().decode())
            except Exception:  # noqa: BLE001
                detail = {"error": "unparseable"}
            return e.code, detail, (time.perf_counter() - started) * 1000
        except Exception as e:  # noqa: BLE001
            return 0, {"error": repr(e)}, (time.perf_counter() - started) * 1000

    @staticmethod
    def is_anthropic(model: str) -> bool:
        return model.startswith("anthropic.")

    def complete(self, model: str, prompt: str, max_tokens: int = 512,
                 temperature: float = 0.0) -> tuple[int, str | None, dict, float]:
        """One-shot completion. Returns (status, text, usage, latency_ms)."""
        if self.is_anthropic(model):
            status, data, ms = self._post("/anthropic/v1/messages", {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
            })
            if status != 200:
                return status, None, data, ms
            blocks = data.get("content") or []
            text = "".join(b.get("text", "") for b in blocks if isinstance(b, dict))
            usage = data.get("usage") or {}
            # normalize to the OpenAI-style names the ledger expects
            return status, text, {
                "prompt_tokens": usage.get("input_tokens"),
                "completion_tokens": usage.get("output_tokens"),
            }, ms

        status, data, ms = self._post("/v1/chat/completions", {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        })
        if status != 200:
            return status, None, data, ms
        text = (data.get("choices") or [{}])[0].get("message", {}).get("content")
        return status, text, data.get("usage") or {}, ms
