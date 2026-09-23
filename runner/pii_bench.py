#!/usr/bin/env python3
"""Compare PII detectors, then prove the round trip through the live gateway.

Two questions, measured separately:

  detect   On prompts with known personal data, what does each detector find?
           A regex knows formats -- an address, a card number, an SSN. It does
           not know that "Priya Raman" is a name, because nothing about the
           string says so. Amazon Comprehend does. The gap is the reason to pay
           for a network call.

  restore  Through the gateway: the model must never see the original, and the
           caller must never see the placeholder.

    python3 runner/pii_bench.py --detect
    python3 runner/pii_bench.py --restore --detector regex
"""
import argparse
import datetime as dt
import json
import pathlib
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from gateway.plugins import pii as PII  # noqa: E402
from gwclient import Client, _session  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODEL = "mistral.ministral-3-3b-instruct"

# Each case lists the substrings a detector must find to be credited. The
# formats are the easy half; the names, addresses and ages are the half that
# tells you whether you needed a model.
CASES = [
    {"text": "Email the invoice to priya.raman@northwind.example and cc accounts@northwind.example.",
     "expect": ["priya.raman@northwind.example", "accounts@northwind.example"]},
    {"text": "Priya Raman called about her order; call her back on (415) 555-0173.",
     "expect": ["Priya Raman", "(415) 555-0173"]},
    {"text": "Customer SSN is 123-45-6789 and the card on file ends 4111 1111 1111 1111.",
     "expect": ["123-45-6789", "4111 1111 1111 1111"]},
    {"text": "Ship to 1600 Amphitheatre Parkway, Mountain View, CA 94043 by Friday.",
     "expect": ["1600 Amphitheatre Parkway", "94043"]},
    {"text": "Dr. Alejandro Núñez reviewed the chart for a 47-year-old patient in Seattle.",
     "expect": ["Alejandro Núñez", "47"]},
    {"text": "The request came from 203.0.113.42 using account bfeeny@example.org.",
     "expect": ["203.0.113.42", "bfeeny@example.org"]},
    {"text": "Wire the retainer to account 000123456789, routing 021000021, attn Marcus Feld.",
     "expect": ["000123456789", "021000021", "Marcus Feld"]},
    {"text": "Summarize this ticket: Jian Wu at Globex says the portal is down since Tuesday.",
     "expect": ["Jian Wu"]},
]


def found(text: str, spans: list[dict]) -> list[str]:
    return [text[s["start"]:s["end"]] for s in spans]


def credited(expect: list[str], hits: list[str]) -> int:
    """An expectation counts as caught when some span covers it."""
    return sum(any(e in h or h in e for h in hits) for e in expect)


def detect_arm(detector: str) -> dict:
    rows, lat = [], []
    total_expected = total_caught = 0
    for case in CASES:
        t0 = time.perf_counter()
        if detector == "regex":
            spans = PII._spans_regex(case["text"], None)
        else:
            spans = PII._spans_comprehend(case["text"], 0.9)
        lat.append((time.perf_counter() - t0) * 1000)
        hits = found(case["text"], spans)
        caught = credited(case["expect"], hits)
        total_expected += len(case["expect"])
        total_caught += caught
        rows.append({"text": case["text"], "expected": case["expect"], "found": hits,
                     "caught": caught, "of": len(case["expect"]),
                     "types": sorted({s["type"] for s in spans})})
        print(f"  {caught}/{len(case['expect'])}  {case['text'][:56]}...")
        for miss in [e for e in case["expect"] if not any(e in h or h in e for h in hits)]:
            print(f"        missed: {miss!r}")
    return {"detector": detector, "caught": total_caught, "expected": total_expected,
            "recall": round(total_caught / total_expected, 3),
            "median_ms": round(statistics.median(lat), 2), "rows": rows}


def configure(session, param: str, detector: str, ttl: float) -> str:
    ssm = session.client("ssm")
    before = ssm.get_parameter(Name=param)["Parameter"]["Value"]
    cfg = json.loads(before)
    plugins = [p for p in cfg["plugins"] if p["plugin"] not in ("pii", "cache", "semantic_cache")]
    at = next((i for i, p in enumerate(plugins) if p["plugin"] == "tenant"), -1) + 1
    plugins.insert(at, {"plugin": "pii", "params": {"detector": detector, "restore": True}})
    cfg["plugins"] = plugins
    ssm.put_parameter(Name=param, Value=json.dumps(cfg), Overwrite=True, Type="String")
    print(f"pii detector={detector}, caches out of the chain; waiting {ttl + 10:.0f}s")
    time.sleep(ttl + 10)
    return before


def restore_arm(stack: str, detector: str, ttl: float) -> dict:
    gw = Client.for_stack(stack)
    restore = configure(gw.session, gw.outputs["PipelineConfigParameter"], detector, ttl)
    rows = []
    try:
        for case in CASES[:5]:
            prompt = (f"Repeat the following line back exactly, with no commentary: {case['text']}")
            status, _, raw, ms = gw.post(
                {"model": f"mantle/{MODEL}", "max_tokens": 120, "temperature": 0,
                 "messages": [{"role": "user", "content": prompt}]},
                {"x-tenant-id": "pii-bench"})
            body = json.loads(raw) if raw else {}
            answer = ""
            if body.get("choices"):
                answer = (body["choices"][0].get("message") or {}).get("content") or ""
            rows.append({
                "expect": case["expect"], "status": status, "ms": round(ms, 1),
                "answer": answer[:300],
                # The model was asked to echo. If the original comes back, the
                # placeholder was substituted on the way out.
                "restored": [e for e in case["expect"] if e in answer],
                "placeholder_leaked": "{" in answer and "}" in answer,
            })
            print(f"  {status}  restored {len(rows[-1]['restored'])}/{len(case['expect'])}"
                  f"  leaked_placeholder={rows[-1]['placeholder_leaked']}")
    finally:
        gw.session.client("ssm").put_parameter(
            Name=gw.outputs["PipelineConfigParameter"], Value=restore, Overwrite=True, Type="String")
    return {"detector": detector, "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", default="gwpipeline")
    ap.add_argument("--detect", action="store_true")
    ap.add_argument("--restore", action="store_true")
    ap.add_argument("--detector", default="comprehend", choices=["comprehend", "regex"])
    ap.add_argument("--ttl", type=float, default=30)
    args = ap.parse_args()

    out = {"window": dt.datetime.now(dt.timezone.utc).isoformat()}
    if args.detect:
        for d in ("regex", "comprehend"):
            print(f"\n{d}:")
            out[d] = detect_arm(d)
            print(f"  recall {out[d]['recall']}  median {out[d]['median_ms']} ms")
    if args.restore:
        print(f"\nround trip, detector={args.detector}:")
        out[f"restore_{args.detector}"] = restore_arm(args.stack, args.detector, args.ttl)

    path = ROOT / "results" / "reports" / ("pii-detect.json" if args.detect
                                           else f"pii-restore-{args.detector}.json")
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
