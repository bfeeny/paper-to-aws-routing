#!/usr/bin/env python3
"""Fetch the MT-Bench question set and record its provenance.

The prompts aren't vendored: they're downloaded from the FastChat repository
(Apache-2.0) and pinned by SHA-256, so a reproducer can verify they got the same
file we did. Writes experiments/prompts/mt_bench.jsonl and a provenance record.

    python3 experiments/fetch_prompts.py
"""

import datetime as dt
import hashlib
import json
import pathlib
import urllib.request

SOURCE = (
    "https://raw.githubusercontent.com/lm-sys/FastChat/main/"
    "fastchat/llm_judge/data/mt_bench/question.jsonl"
)
OUT_DIR = pathlib.Path(__file__).parent / "prompts"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = urllib.request.urlopen(SOURCE, timeout=60).read()
    digest = hashlib.sha256(raw).hexdigest()

    rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    # MT-Bench turns are multi-turn; this study uses the first turn only, which is
    # what the routing decision sees.
    prompts = [
        {
            "id": f"mtbench-{r['question_id']}",
            "category": r.get("category"),
            "prompt": r["turns"][0],
        }
        for r in rows
    ]

    out = OUT_DIR / "mt_bench.jsonl"
    with out.open("w") as fh:
        for p in prompts:
            fh.write(json.dumps(p) + "\n")

    provenance = {
        "source": SOURCE,
        "sha256": digest,
        "retrieved": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "license": "Apache-2.0 (FastChat)",
        "records": len(prompts),
        "note": "first turn only",
    }
    (OUT_DIR / "mt_bench.provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    print(f"{len(prompts)} prompts -> {out}")
    print(f"sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
