#!/usr/bin/env python3
"""Build routing labels from Chatbot Arena human preference battles.

RouteLLM's insight is that a routing label already exists in preference data: when
a strong model and a weak model answer the same prompt and humans pick a winner,
the strong model's win says "this prompt needed the strong model."

    strong wins            -> 1  route to strong
    weak wins, or a tie    -> 0  the weak model sufficed

Only battles between a model in the strong tier and one in the weak tier are used;
same-tier battles carry no routing signal. The tier map below is explicit on
purpose — it is the single most consequential judgment in this pipeline, and it
should be arguable rather than buried.

    python3 experiments/fetch_arena.py [--limit 8000]
"""

import argparse
import csv
import hashlib
import json
import pathlib
import random
import sys
import urllib.request

csv.field_size_limit(10**9)

ROOT = pathlib.Path(__file__).resolve().parent.parent
SOURCE = ("https://huggingface.co/datasets/lmarena-ai/arena-human-preference-55k/"
          "resolve/main/train.csv")
CACHE = ROOT / "experiments" / "cache" / "arena-55k.csv"
OUT = ROOT / "experiments" / "arena_labels.jsonl"

# Frontier-class models of the era this data covers.
STRONG = {
    "gpt-4-1106-preview", "gpt-4-0613", "gpt-4-0314", "gpt-4-turbo-2024-04-09",
    "gpt-4-0125-preview", "claude-2.1", "claude-2.0", "claude-1",
    "gemini-pro-dev-api", "mistral-medium", "claude-3-opus-20240229",
    "gpt-4", "bard-jan-24-gemini-pro",
}
# Small, open, or older models — the cheap tier a router would prefer.
WEAK = {
    "mixtral-8x7b-instruct-v0.1", "mistral-7b-instruct", "mistral-7b-instruct-v0.2",
    "llama-2-70b-chat", "llama-2-13b-chat", "llama-2-7b-chat",
    "vicuna-33b", "vicuna-13b", "vicuna-7b", "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-1106", "gpt-3.5-turbo-0314", "gpt-3.5-turbo-0125",
    "claude-instant-1", "zephyr-7b-beta", "openchat-3.5", "starling-lm-7b-alpha",
    "qwen1.5-7b-chat", "qwen1.5-14b-chat", "tulu-2-dpo-70b", "yi-34b-chat",
    "openhermes-2.5-mistral-7b", "wizardlm-13b", "koala-13b", "chatglm3-6b",
    "gemma-7b-it", "gemma-2b-it", "pplx-7b-online", "solar-10.7b-instruct-v1.0",
}


def download() -> bytes:
    if CACHE.exists():
        print(f"using cached {CACHE.relative_to(ROOT)}")
        return CACHE.read_bytes()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {SOURCE}")
    raw = urllib.request.urlopen(SOURCE, timeout=600).read()
    CACHE.write_bytes(raw)
    return raw


def first_turn(prompt_field: str) -> str | None:
    try:
        turns = json.loads(prompt_field)
    except Exception:  # noqa: BLE001
        return None
    if isinstance(turns, list) and turns and isinstance(turns[0], str):
        return turns[0].strip() or None
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=8000,
                    help="cap on labeled examples (seeded sample)")
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--min-chars", type=int, default=12)
    args = ap.parse_args()

    raw = download()
    digest = hashlib.sha256(raw).hexdigest()

    kept, skipped_tier, skipped_prompt = [], 0, 0
    reader = csv.DictReader(raw.decode("utf-8", "replace").splitlines())
    for row in reader:
        a, b = row.get("model_a", ""), row.get("model_b", "")
        if a in STRONG and b in WEAK:
            strong_is_a = True
        elif b in STRONG and a in WEAK:
            strong_is_a = False
        else:
            skipped_tier += 1
            continue

        prompt = first_turn(row.get("prompt", ""))
        if not prompt or len(prompt) < args.min_chars:
            skipped_prompt += 1
            continue

        tie = row.get("winner_tie") == "1"
        a_won = row.get("winner_model_a") == "1"
        strong_won = (not tie) and (a_won if strong_is_a else not a_won)

        kept.append({
            "id": row.get("id"),
            "prompt": prompt[:8000],
            "label": 1 if strong_won else 0,
            "strong_model": a if strong_is_a else b,
            "weak_model": b if strong_is_a else a,
            "tie": tie,
        })

    rng = random.Random(args.seed)
    rng.shuffle(kept)
    sample = kept[: args.limit] if args.limit else kept

    OUT.write_text("".join(json.dumps(r) + "\n" for r in sample))
    prov = {
        "source": SOURCE, "sha256": digest,
        "battles_total": skipped_tier + skipped_prompt + len(kept),
        "cross_tier_battles": len(kept),
        "written": len(sample), "seed": args.seed,
        "strong_tier": sorted(STRONG), "weak_tier": sorted(WEAK),
        "label_rule": "1 = strong model won; 0 = weak model won or tie",
    }
    (OUT.with_suffix(".provenance.json")).write_text(json.dumps(prov, indent=2) + "\n")

    pos = sum(r["label"] for r in sample)
    ties = sum(r["tie"] for r in sample)
    print(f"cross-tier battles: {len(kept)}  (skipped: {skipped_tier} same-tier/unknown, "
          f"{skipped_prompt} unusable prompts)")
    print(f"written: {len(sample)}  strong-needed: {pos} ({pos/len(sample):.1%})  "
          f"ties counted as weak-sufficed: {ties}")
    print(f"sha256 {digest}")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
