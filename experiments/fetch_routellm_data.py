#!/usr/bin/env python3
"""Fetch RouteLLM's own published training data and convert it to routing labels.

The paper's results come from *augmented* preference data, not raw Arena battles,
and the authors published it:

    routellm/gpt4_judge_battles   109,101 GPT-4-judged battles
    routellm/mmlu_battles           1,531 MMLU-derived battles

Using their labels with our features isolates one variable. If a router trained on
this data works, the earlier null result was about label scale. If it still fails,
the limitation is the representation — Titan embeddings plus a linear model versus
their matrix-factorisation / BERT / causal-LLM architectures. Either answer is
worth having, and it costs one experiment instead of a week of label manufacture.

Note the portability gap this exposes: their released routers cannot be dropped
into a Bedrock stack, because `mf` and `bert` consume OpenAI
`text-embedding-3-small` features. The labels port; the models do not.

Requires the data-prep venv (pyarrow):  .venv/bin/python experiments/fetch_routellm_data.py
"""

import argparse
import hashlib
import json
import pathlib
import random
import urllib.request

import pyarrow.parquet as pq

ROOT = pathlib.Path(__file__).resolve().parent.parent
CACHE = ROOT / "experiments" / "cache"
SOURCES = {
    "gpt4_judge": ("routellm/gpt4_judge_battles",
                   "https://huggingface.co/api/datasets/routellm/gpt4_judge_battles/"
                   "parquet/default/train/0.parquet"),
    "mmlu": ("routellm/mmlu_battles",
             "https://huggingface.co/api/datasets/routellm/mmlu_battles/"
             "parquet/default/train/0.parquet"),
}


def grab(name: str, url: str) -> pathlib.Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"routellm-{name}.parquet"
    if path.exists():
        print(f"  cached {path.name} ({path.stat().st_size / 1e6:.1f} MB)")
        return path
    print(f"  downloading {url}")
    raw = urllib.request.urlopen(url, timeout=900).read()
    path.write_bytes(raw)
    print(f"  wrote {path.name} ({len(raw) / 1e6:.1f} MB), "
          f"sha256 {hashlib.sha256(raw).hexdigest()[:16]}…")
    return path


def to_labels(path: pathlib.Path, source: str) -> list[dict]:
    table = pq.read_table(path).to_pylist()
    out = []
    for r in table:
        prompt = r.get("prompt")
        if isinstance(prompt, str) and prompt.startswith("["):
            try:
                turns = json.loads(prompt)
                prompt = turns[0] if isinstance(turns, list) and turns else None
            except Exception:  # noqa: BLE001
                pass
        if not isinstance(prompt, str) or len(prompt.strip()) < 12:
            continue

        # model_a is the strong model (GPT-4) throughout these sets; keep the
        # label definition identical to our Arena pipeline.
        a_won = bool(r.get("winner_model_a"))
        tie = bool(r.get("winner_tie"))
        out.append({
            "id": str(r.get("id")),
            "source": source,
            "prompt": prompt.strip()[:8000],
            "label": 1 if (a_won and not tie) else 0,
            "strong_model": r.get("model_a"),
            "weak_model": r.get("model_b"),
            "tie": tie,
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = use everything")
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--out", default="experiments/routellm_labels.jsonl")
    args = ap.parse_args()

    rows = []
    for name, (dataset, url) in SOURCES.items():
        print(f"{dataset}:")
        rows += to_labels(grab(name, url), dataset)

    random.Random(args.seed).shuffle(rows)
    if args.limit:
        rows = rows[: args.limit]

    out = ROOT / args.out
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))

    pos = sum(r["label"] for r in rows)
    ties = sum(r["tie"] for r in rows)
    by_source = {}
    for r in rows:
        by_source[r["source"]] = by_source.get(r["source"], 0) + 1
    print(f"\nwrote {len(rows)} labels to {out.relative_to(ROOT)}")
    print(f"  strong-needed: {pos} ({pos / len(rows):.1%})   ties (→ weak sufficed): {ties}")
    print(f"  by source: {by_source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
