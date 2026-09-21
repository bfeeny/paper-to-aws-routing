#!/usr/bin/env python3
"""Train the routing scorer on Chatbot Arena preference data.

Features are Titan v2 embeddings (256-d) of the prompt; the model is L2-regularised
logistic regression fitted by gradient descent and shipped as JSON weights, so the
interceptor needs no numpy and no container image.

Two evaluations, because they answer different questions:

  * a held-out slice of Arena — does the model fit its own distribution?
  * our MT-Bench judged pairs — does it transfer to the prompts and the model pair
    the study actually routes? This is the honest test, and the harder one.

Embeddings are cached by content hash, so re-fitting costs nothing.

    python3 analysis/train_router.py --labels experiments/arena_labels.jsonl \
        --external results/judgments/<dir>
"""

import argparse
import concurrent.futures as cf
import hashlib
import json
import pathlib
import urllib.error
import urllib.request

import botocore.session
import numpy as np
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

ROOT = pathlib.Path(__file__).resolve().parent.parent
EMBED_MODEL = "amazon.titan-embed-text-v2:0"
DIMS = 256  # overridable with --dims; Titan v2 supports 256/512/1024
CACHE_DIR = ROOT / "experiments" / "cache"


def _embed_one(creds, region, text, dims, attempts=5):
    url = f"https://bedrock-runtime.{region}.amazonaws.com/model/{EMBED_MODEL}/invoke"
    body = json.dumps({"inputText": text[:8000], "dimensions": dims,
                       "normalize": True}).encode()
    for attempt in range(attempts):
        req = AWSRequest(method="POST", url=url, data=body,
                         headers={"Content-Type": "application/json"})
        SigV4Auth(creds, "bedrock", region).add_auth(req)
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, data=body, headers=dict(req.headers)),
                timeout=60,
            ) as r:
                return json.load(r)["embedding"]
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and attempt < attempts - 1:
                import time
                time.sleep(1.5 * (attempt + 1))  # throttling: back off and retry
                continue
            raise
    raise RuntimeError("embedding failed after retries")


def embed(texts: list[str], profile: str, region: str, dims: int = DIMS,
          workers: int = 8) -> np.ndarray:
    key = hashlib.sha256(("|".join(texts)).encode()).hexdigest()[:16]
    cache = CACHE_DIR / f"emb-{EMBED_MODEL.replace(':', '_')}-{dims}-{key}.npy"
    if cache.exists():
        print(f"  embeddings from cache ({cache.name})")
        return np.load(cache)

    creds = botocore.session.Session(profile=profile).get_credentials().get_frozen_credentials()
    out: list[list[float] | None] = [None] * len(texts)
    done = 0
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_embed_one, creds, region, t, dims): i
                   for i, t in enumerate(texts)}
        for fut in cf.as_completed(futures):
            out[futures[fut]] = fut.result()
            done += 1
            if done % 500 == 0:
                print(f"  embedded {done}/{len(texts)}")
    arr = np.array(out, dtype=float)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache, arr)
    return arr


def fit_logistic(X, y, l2=1.0, epochs=3000, lr=1.0, seed=20260921):
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 0.01, X.shape[1])
    b, n = 0.0, len(y)
    for _ in range(epochs):
        p = 1 / (1 + np.exp(-(X @ w + b)))
        w -= lr * (X.T @ (p - y) / n + l2 * w / n)
        b -= lr * float(np.sum(p - y) / n)
    return w, b


def scores(X, w, b):
    return 1 / (1 + np.exp(-(X @ w + b)))


def evaluate(X, y, w, b) -> dict:
    p = scores(X, w, b)
    acc = float(((p >= 0.5).astype(int) == y).mean())
    majority = float(max(y.mean(), 1 - y.mean()))
    # AUC via rank statistic: robust to threshold choice, unlike accuracy
    order = np.argsort(p)
    ranks = np.empty(len(p), dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    n1, n0 = float(y.sum()), float((1 - y).sum())
    auc = float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else None
    return {
        "n": int(len(y)),
        "accuracy": round(acc, 3),
        "majority_baseline": round(majority, 3),
        "lift_over_majority": round(acc - majority, 3),
        "auc": round(auc, 3) if auc is not None else None,
        "score_min": round(float(p.min()), 3),
        "score_max": round(float(p.max()), 3),
        "score_spread": round(float(p.max() - p.min()), 3),
        "score_std": round(float(p.std()), 3),
    }


def external_labels(judgment_dir: pathlib.Path, prompts_file: pathlib.Path):
    prompts = {json.loads(l)["id"]: json.loads(l)["prompt"]
               for l in prompts_file.read_text().splitlines() if l.strip()}
    texts, ys = [], []
    for line in (judgment_dir / "judgments.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        j = json.loads(line)
        if j["outcome"] == "baseline_win":
            y = 1
        elif j["outcome"] in ("candidate_win", "tie"):
            y = 0
        else:
            continue
        texts.append(prompts[j["prompt_id"]])
        ys.append(y)
    return texts, np.array(ys, dtype=float)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="experiments/arena_labels.jsonl")
    ap.add_argument("--external", default=None,
                    help="judgment dir used as an out-of-distribution test set")
    ap.add_argument("--experiment", default="experiments/mistral-gap.json")
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--dims", type=int, default=DIMS, choices=[256, 512, 1024])
    ap.add_argument("--holdout-frac", type=float, default=0.15)
    ap.add_argument("--profile", default="personal")
    ap.add_argument("--region", default="us-east-1")
    args = ap.parse_args()

    rows = [json.loads(l) for l in (ROOT / args.labels).read_text().splitlines() if l.strip()]
    print(f"labels: {len(rows)} ({sum(r['label'] for r in rows) / len(rows):.1%} strong-needed)")

    X = embed([r["prompt"] for r in rows], args.profile, args.region, args.dims)
    y = np.array([r["label"] for r in rows], dtype=float)

    cut = int(len(rows) * (1 - args.holdout_frac))
    Xtr, ytr, Xte, yte = X[:cut], y[:cut], X[cut:], y[cut:]
    w, b = fit_logistic(Xtr, ytr, l2=args.l2)

    report = {"train": evaluate(Xtr, ytr, w, b), "arena_holdout": evaluate(Xte, yte, w, b)}

    if args.external:
        cfg = json.loads((ROOT / args.experiment).read_text())
        texts, ys = external_labels(ROOT / args.external, ROOT / cfg["prompts"])
        if len(ys):
            Xext = embed(texts, args.profile, args.region, args.dims)
            report["external_mtbench"] = evaluate(Xext, ys, w, b)

    artifact = {
        "model": "logistic_regression",
        "embedding_model": EMBED_MODEL,
        "dims": args.dims,
        "l2": args.l2,
        "weights": [round(float(x), 6) for x in w],
        "bias": round(float(b), 6),
        "trained_on": args.labels,
        "train_rows": int(cut),
        "label_rule": "1 = strong model needed; 0 = weak sufficed (win or tie)",
        "report": report,
    }
    out = ROOT / "router" / "artifacts" / "router_weights.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2) + "\n")

    print(json.dumps(report, indent=2))
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
