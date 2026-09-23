"""Keep personal data out of the model, and put it back in the answer.

Three different things get called "PII handling" at a gateway, and they are not
substitutes:

  block     refuse the request outright. Bedrock Guardrails does this natively
            and the `guardrail` plugin already wires it up. Safe, and useless
            for the common case where the personal data is the *point* of the
            request ("summarize this support ticket").
  redact    replace the personal data with a placeholder and send that. The
            model never sees it. The answer comes back talking about
            {NAME_0}, which is correct and unreadable.
  tokenize  redact on the way in, and substitute the real values back into the
            answer on the way out. The model never sees the data, the caller
            never sees a placeholder.

Only the third is transparent to the caller, and it is only possible at a
gateway that can see both halves of the call. It needs the mapping from
placeholder to original to survive from the request phase to the response
phase -- and the RESPONSE interceptor receives no request, so the mapping
travels through the correlation record like everything else.

That mapping is the sharp edge. It is a list of exactly the strings you were
trying to protect, and this plugin writes it to DynamoDB. It is therefore
written only when `restore` is on, carries a short TTL, and belongs in a table
with a customer-managed key in any real deployment. Redacting without restoring
needs no such record, and when you can live with placeholders in the answer,
that is the safer configuration. The choice is a real one, not an oversight.

`screen: true` puts each prompt to the much cheaper `ContainsPiiEntities`
first and only locates spans when that says there is something to locate. It
is off by default because measuring it did not support the idea: on five
prompts with obvious personal data, the screen reported *no labels at all* for
an email address, and for a bank account number alongside a name, both of which
`DetectPiiEntities` found with score 1.0. That is not a threshold that wants
lowering -- the cheap call answers a weaker question and answers it wrong here.
Used as a gate, it would forward exactly the data this plugin exists to hide,
silently. Screen only where a false negative is survivable; never in front of
masking.

Detection uses Amazon Comprehend's `DetectPiiEntities`, which returns typed
spans with offsets and confidence -- what you need to substitute precisely.
Guardrails answers a different question ("does this violate policy"), and a
regex answers a third ("does this look like a credit card"). The `detector`
parameter picks one; `regex` needs no network call and is the fallback when
Comprehend is unavailable in-Region, at the cost of catching only the formats
it knows.
"""
import json
import os
import re

from ..pipeline import Call, Plugin, register

_comprehend = None

# Deliberately narrow: formats with enough structure to match without guessing.
# Names, addresses and anything contextual are Comprehend's job, not a regex's.
PATTERNS = {
    "EMAIL": re.compile(r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b"),
    "PHONE": re.compile(r"\b(?:\+?1[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b"),
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "CREDIT_DEBIT_NUMBER": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    "IP_ADDRESS": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}


def _client():
    global _comprehend
    if _comprehend is None:
        import boto3
        _comprehend = boto3.client("comprehend")
    return _comprehend


def _contains_pii(text: str, min_score: float) -> bool:
    """A 50x cheaper question: is there anything in here at all?

    ContainsPiiEntities is billed at $0.000002 per 100-character unit against
    DetectPiiEntities' $0.0001, both with a three-unit minimum -- $0.000006 a
    call against $0.0003. The arithmetic is attractive and the behavior is not:
    see the module docstring. Kept because it is the right tool for deciding
    whether to *alert* on a prompt, which is a question you are allowed to get
    wrong occasionally.
    """
    r = _client().contains_pii_entities(Text=text[:5000], LanguageCode="en")
    return any(label["Score"] >= min_score for label in r.get("Labels", []))


def _spans_comprehend(text: str, min_score: float) -> list[dict]:
    r = _client().detect_pii_entities(Text=text[:5000], LanguageCode="en")
    return [{"start": e["BeginOffset"], "end": e["EndOffset"], "type": e["Type"],
             "score": e["Score"]}
            for e in r.get("Entities", []) if e["Score"] >= min_score]


def _spans_regex(text: str, types: list[str] | None) -> list[dict]:
    out = []
    for name, pattern in PATTERNS.items():
        if types and name not in types:
            continue
        for m in pattern.finditer(text):
            out.append({"start": m.start(), "end": m.end(), "type": name, "score": 1.0})
    return out


def mask(text: str, spans: list[dict]) -> tuple[str, dict]:
    """Replace spans right to left so earlier offsets stay valid.

    The same original value always gets the same placeholder, so a prompt that
    mentions one person twice does not become a prompt about two people.
    """
    mapping, counter, seen = {}, {}, {}
    # Longest first at the same start, so overlapping detections do not nest.
    spans = sorted(spans, key=lambda s: (s["start"], -(s["end"] - s["start"])))
    kept, last_end = [], -1
    for s in spans:
        if s["start"] >= last_end:
            kept.append(s)
            last_end = s["end"]
    for s in reversed(kept):
        original = text[s["start"]:s["end"]]
        if original in seen:
            token = seen[original]
        else:
            n = counter.get(s["type"], 0)
            counter[s["type"]] = n + 1
            token = f"{{{s['type']}_{n}}}"
            seen[original] = token
            mapping[token] = original
        text = text[:s["start"]] + token + text[s["end"]:]
    return text, mapping


def unmask(text: str, mapping: dict) -> str:
    for token, original in mapping.items():
        text = text.replace(token, original)
    return text


@register
class Pii(Plugin):
    name = "pii"

    @property
    def needs_response(self) -> bool:
        return bool(self.params.get("restore", True))

    def _detect(self, text: str) -> list[dict]:
        if self.params.get("detector", "comprehend") == "regex":
            return _spans_regex(text, self.params.get("types"))
        min_score = float(self.params.get("min_score", 0.9))
        # Off by default, and deliberately so -- a screen that misses is a leak.
        if self.params.get("screen") and not _contains_pii(text, min_score):
            return []
        return _spans_comprehend(text, min_score)

    def on_request(self, call: Call) -> None:
        messages = call.body.get("messages")
        if not isinstance(messages, list):
            return None
        mapping: dict = {}
        found: list[str] = []
        for m in messages:
            content = m.get("content")
            if not isinstance(content, str) or not content.strip():
                continue
            spans = self._detect(content)
            if not spans:
                continue
            masked, part = mask(content, spans)
            m["content"] = masked
            mapping.update(part)
            found.extend(s["type"] for s in spans)

        call.attrs["pii_found"] = len(found)
        call.attrs["pii_types"] = ",".join(sorted(set(found))) or None
        if not mapping:
            return None
        if self.params.get("restore", True):
            # The one place the originals are persisted. Short TTL, and only
            # because the answer has to be readable.
            call.attrs.setdefault("remember", {})["pii_map"] = json.dumps(mapping)
        call.attrs["pii_masked"] = len(mapping)
        call.attrs["pii_map_live"] = mapping     # for a short-circuit, below
        return None

    def on_abort(self, call: Call, verdict) -> None:
        """Re-personalize an answer served without a model call.

        Placing this plugin before the cache means the cache key is computed on
        the masked prompt, so nothing personal is ever stored -- and two
        different people asking the same question mask to the same text and
        share an entry, which raises the hit rate rather than lowering it.

        The cost is that the stored answer is full of placeholders. A cache hit
        short-circuits the chain, so `on_response` never runs and the caller
        would receive `{EMAIL_0}` verbatim. The substitution has to happen here,
        against the mapping built from *this* request -- which is the right
        mapping precisely because the masked prompts matched.
        """
        mapping = call.attrs.get("pii_map_live")
        body = getattr(verdict, "body", None)
        if not mapping or not isinstance(body, dict) or not body.get("choices"):
            return
        restored = 0
        for choice in body["choices"]:
            message = choice.get("message") or {}
            content = message.get("content")
            if isinstance(content, str):
                new = unmask(content, mapping)
                restored += new != content
                message["content"] = new
        call.attrs["pii_restored_on_serve"] = restored

    def on_response(self, call: Call) -> None:
        raw = call.attrs.get("recalled", {}).get("pii_map")
        if not raw:
            return
        mapping = json.loads(raw)
        body = call.attrs.get("replacement_body") or call.response
        if not body or not body.get("choices"):
            return
        restored = 0
        for choice in body["choices"]:
            message = choice.get("message") or {}
            content = message.get("content")
            if isinstance(content, str):
                new = unmask(content, mapping)
                restored += new != content
                message["content"] = new
        call.attrs["pii_restored"] = restored
