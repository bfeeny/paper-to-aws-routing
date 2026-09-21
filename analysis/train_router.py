#!/usr/bin/env python3
"""Train the routing scorer from judged pairs.

Label definition: for each prompt we already know whether the weak model's answer
was good enough, because the pairwise judge told us.

    baseline_win                 -> 1  the strong model was needed
    candidate_win | tie          -> 0  the weak model sufficed
    inconsistent                 ->    dropped; the judge contradicted itself

Features are Titan v2 embeddings of the prompt. The model is L2-regularised
logistic regression fitted with plain gradient descent — small enough to ship as
JSON weights and score in a Lambda without numpy.

Trains on the dev split only, so the held-out prompts stay unseen.

    python3 analysis/train_router.py --judgment results/judgments/<dir>
"""

import argparse
import json
import pathlib
import random

import urllib.request

import botocore.session
import numpy as np
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

ROOT = pathlib.Path(__file__).resolve().parent.parent
EMBED_MODEL = "amazon.titan-embed-text-v2:0"
DIMS = 256


def embed_all(texts: list[str], profile: str, region: str) -> np.ndarray:
    """Signed HTTP rather than a boto3 client: the local botocore predates the
    bedrock-runtime service model, while the signing primitives work fine."""
    creds = botocore.session.Session(profile=profile).get_credentials().get_frozen_credentials()
    url = f"https://bedrock-runtime.{region}.amazonaws.com/model/{EMBED_MODEL}/invoke"
    out = []
    for i, t in enumerate(texts, 1):
        body = json.dumps({"inputText": t[:8000], "dimensions": DIMS, "normalize": True}).encode()
        req = AWSRequest(method="POST", url=url, data=body,
                         headers={"Content-Type": "application/json"})
        SigV4Auth(creds, "bedrock", region).add_auth(req)
        with urllib.request.urlopen(
            urllib.request.Request(url, data=body, headers=dict(req.headers)), timeout=60
        ) as r:
            out.append(json.load(r)["embedding"])
        if i % 20 == 0:
            print(f"  embedded {i}/{len(texts)}")
    return np.array(out, dtype=float)


def fit_logistic(X, y, l2=1.0, epochs=4000, lr=0.5, seed=20260921):
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 0.01, X.shape[1])
    b = 0.0
    n = len(y)
    for _ in range(epochs):
        z = X @ w + b
        p = 1 / (1 + np.exp(-z))
        gw = X.T @ (p - y) / n + l2 * w / n
        gb = float(np.sum(p - y) / n)
        w -= lr * gw
        b -= lr * gb
    return w, b


def accuracy(X, y, w, b, thr=0.5):
    p = 1 / (1 + np.exp(-(X @ w + b)))
    return float(((p >= thr).astype(int) == y).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judgment", required=True)
    ap.add_argument("--experiment", default="experiments/mistral-gap.json")
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--profile", default="personal")
    ap.add_argument("--region", default="us-east-1")
    args = ap.parse_args()

    cfg = json.loads((ROOT / args.experiment).read_text())
    prompts = {json.loads(l)["id"]: json.loads(l)["prompt"]
               for l in (ROOT / cfg["prompts"]).read_text().splitlines() if l.strip()}

    # dev / held-out split, reproduced exactly as the runner computes it
    rows = [json.loads(l) for l in (ROOT / cfg["prompts"]).read_text().splitlines() if l.strip()]
    rng = random.Random(cfg["split"]["seed"])
    order = list(range(len(rows)))
    rng.shuffle(order)
    cut = int(len(rows) * cfg["split"]["dev_fraction"])
    dev_ids = {rows[i]["id"] for i in order[:cut]}

    jdir = ROOT / args.judgment
    labelled = []
    for line in (jdir / "judgments.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        j = json.loads(line)
        if j["outcome"] == "baseline_win":
            labelled.append((j["prompt_id"], 1))
        elif j["outcome"] in ("candidate_win", "tie"):
            labelled.append((j["prompt_id"], 0))

    train = [(pid, y) for pid, y in labelled if pid in dev_ids]
    test = [(pid, y) for pid, y in labelled if pid not in dev_ids]
    print(f"labelled prompts: {len(labelled)} "
          f"(train/dev {len(train)}, held-out {len(test)}; "
          f"{sum(y for _, y in labelled)} need the strong model)")
    if len(train) < 8:
        print("refusing to train: too few dev labels to fit anything meaningful")
        return 2

    Xtr = embed_all([prompts[p] for p, _ in train], args.profile, args.region)
    ytr = np.array([y for _, y in train], dtype=float)
    w, b = fit_logistic(Xtr, ytr, l2=args.l2)

    report = {"train_n": len(train), "train_accuracy": round(accuracy(Xtr, ytr, w, b), 3)}
    if test:
        Xte = embed_all([prompts[p] for p, _ in test], args.profile, args.region)
        yte = np.array([y for _, y in test], dtype=float)
        report["heldout_n"] = len(test)
        report["heldout_accuracy"] = round(accuracy(Xte, yte, w, b), 3)
        report["heldout_base_rate"] = round(float(yte.mean()), 3)
        report["heldout_majority_baseline"] = round(max(yte.mean(), 1 - yte.mean()), 3)

    artifact = {
        "model": "logistic_regression",
        "embedding_model": EMBED_MODEL,
        "dims": DIMS,
        "l2": args.l2,
        "weights": [round(float(x), 6) for x in w],
        "bias": round(float(b), 6),
        "trained_on": args.judgment,
        "label_rule": "1 = strong model was needed (baseline_win); 0 = weak sufficed (win or tie)",
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
