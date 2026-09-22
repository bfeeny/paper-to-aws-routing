#!/usr/bin/env python3
"""Train the routing scorer on Chatbot Arena preference data.

Features are Titan v2 embeddings (256-d) of the prompt; the model is L2-regularized
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
import http.client
import json
import pathlib
import shutil
import ssl
import threading
import time

import botocore.session
import numpy as np
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

ROOT = pathlib.Path(__file__).resolve().parent.parent
EMBED_MODEL = "amazon.titan-embed-text-v2:0"
DIMS = 256  # overridable with --dims; Titan v2 supports 256/512/1024
CACHE_DIR = ROOT / "experiments" / "cache"

# One TLS context for the process, one kept-alive connection per worker thread.
#
# urlopen() opens a fresh HTTPS connection per call, so embedding 110k prompts
# meant 110k TLS handshakes. Past a few threads the cost is not the handshake
# itself but contention on OpenSSL's provider lock: measured at 4.5 embeddings
# per second with sixteen threads pinned at 1500% CPU, nearly all of it inside
# ossl_lib_ctx_get_data. Keep-alive takes the handshake off the hot path.
_SSL_CTX = ssl.create_default_context()
_TLS = threading.local()


def _conn(region: str) -> http.client.HTTPSConnection:
    c = getattr(_TLS, "conn", None)
    if c is None:
        c = http.client.HTTPSConnection(f"bedrock-runtime.{region}.amazonaws.com",
                                        context=_SSL_CTX, timeout=60)
        _TLS.conn = c
    return c


def _drop_conn() -> None:
    c = getattr(_TLS, "conn", None)
    if c is not None:
        try:
            c.close()
        except Exception:  # noqa: BLE001 — a failed close tells us nothing useful
            pass
    _TLS.conn = None


def _embed_one(creds, region, text, dims, attempts=5):
    host = f"bedrock-runtime.{region}.amazonaws.com"
    path = f"/model/{EMBED_MODEL}/invoke"
    body = json.dumps({"inputText": text[:8000], "dimensions": dims,
                       "normalize": True}).encode()
    for attempt in range(attempts):
        req = AWSRequest(method="POST", url=f"https://{host}{path}", data=body,
                         headers={"Content-Type": "application/json", "Host": host})
        SigV4Auth(creds, "bedrock", region).add_auth(req)
        try:
            c = _conn(region)
            c.request("POST", path, body=body, headers=dict(req.headers))
            resp = c.getresponse()
            payload = resp.read()  # must drain, or the connection can't be reused
            if resp.status == 200:
                return json.loads(payload)["embedding"]
            if resp.status in (429, 503) and attempt < attempts - 1:
                time.sleep(1.5 * (attempt + 1))  # throttling: back off and retry
                continue
            raise RuntimeError(f"embed HTTP {resp.status}: {payload[:200]!r}")
        except (http.client.HTTPException, OSError):
            _drop_conn()  # far end hung up on a pooled connection; rebuild it
            if attempt < attempts - 1:
                time.sleep(0.5 * (attempt + 1))
                continue
            raise
    raise RuntimeError("embedding failed after retries")


def _embed_chunk(creds, region, texts, dims, workers) -> np.ndarray:
    out: list[list[float] | None] = [None] * len(texts)
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_embed_one, creds, region, t, dims): i
                   for i, t in enumerate(texts)}
        for fut in cf.as_completed(futures):
            out[futures[fut]] = fut.result()
    return np.array(out, dtype=float)


def embed(texts: list[str], profile: str, region: str, dims: int = DIMS,
          workers: int = 8, chunk: int = 2000) -> np.ndarray:
    """Embed in chunks, saving each one, so an interrupted run resumes.

    The whole-array cache is the fast path on a re-fit. The per-chunk parts are
    insurance: at 110k prompts a single failure near the end used to discard
    every embedding bought up to that point.
    """
    key = hashlib.sha256(("|".join(texts)).encode()).hexdigest()[:16]
    stem = f"emb-{EMBED_MODEL.replace(':', '_')}-{dims}-{key}"
    cache = CACHE_DIR / f"{stem}.npy"
    if cache.exists():
        print(f"  embeddings from cache ({cache.name})", flush=True)
        return np.load(cache)

    parts_dir = CACHE_DIR / f"{stem}.parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    creds = botocore.session.Session(profile=profile).get_credentials().get_frozen_credentials()

    parts, started = [], time.perf_counter()
    for start in range(0, len(texts), chunk):
        part = parts_dir / f"{start:08d}.npy"
        if part.exists():
            parts.append(np.load(part))
            continue
        arr = _embed_chunk(creds, region, texts[start:start + chunk], dims, workers)
        np.save(part, arr)
        parts.append(arr)
        done = start + len(arr)
        rate = done / max(time.perf_counter() - started, 1e-9)
        print(f"  embedded {done}/{len(texts)}  ({rate:.0f}/s)", flush=True)

    arr = np.concatenate(parts) if parts else np.zeros((0, dims))
    np.save(cache, arr)
    shutil.rmtree(parts_dir, ignore_errors=True)
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
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--profile", default="personal")
    ap.add_argument("--region", default="us-east-1")
    args = ap.parse_args()

    rows = [json.loads(l) for l in (ROOT / args.labels).read_text().splitlines() if l.strip()]
    print(f"labels: {len(rows)} ({sum(r['label'] for r in rows) / len(rows):.1%} strong-needed)")

    X = embed([r["prompt"] for r in rows], args.profile, args.region, args.dims,
              workers=args.workers)
    y = np.array([r["label"] for r in rows], dtype=float)

    cut = int(len(rows) * (1 - args.holdout_frac))
    Xtr, ytr, Xte, yte = X[:cut], y[:cut], X[cut:], y[cut:]
    w, b = fit_logistic(Xtr, ytr, l2=args.l2)

    report = {"train": evaluate(Xtr, ytr, w, b), "arena_holdout": evaluate(Xte, yte, w, b)}

    if args.external:
        cfg = json.loads((ROOT / args.experiment).read_text())
        texts, ys = external_labels(ROOT / args.external, ROOT / cfg["prompts"])
        if len(ys):
            Xext = embed(texts, args.profile, args.region, args.dims, workers=args.workers)
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
