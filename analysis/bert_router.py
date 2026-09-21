#!/usr/bin/env python3
"""Fine-tune an encoder as a router, the arm the frozen-embedding runs could not test.

Every other router here reads a frozen Titan vector. A probe on frozen features
can only use what a general-purpose retrieval embedding already encodes, and
that turned out to be topic -- which is why every in-distribution result
dissolved into group lookup. This arm changes the one variable that matters:
the encoder is trained on the routing objective itself.

It is the closest thing in this study to the paper's BERT router. Two honest
deviations: a smaller backbone by default, and our cascade labels rather than
Arena preferences.

Evaluated exactly like the others -- random split for reference, then
leave-one-task-out, which is the only split that asks whether difficulty
generalises rather than whether the task was memorised.

    .venv/bin/python analysis/bert_router.py --epochs 3
"""
import argparse, json, pathlib, sys, time
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cascade_sweep import auc, auc_ci

ROOT = pathlib.Path(__file__).resolve().parent.parent


class DS(Dataset):
    def __init__(self, texts, labels, tok, maxlen):
        self.enc = tok(texts, truncation=True, max_length=maxlen,
                       padding="max_length", return_tensors="pt")
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return {k: v[i] for k, v in self.enc.items()} | {"labels": self.y[i]}


def train_eval(texts, y, tr_idx, te_idx, model_name, epochs, lr, bs, maxlen, device, seed=0):
    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2).to(device)
    tr = DataLoader(DS([texts[i] for i in tr_idx], y[tr_idx].astype(int), tok, maxlen),
                    batch_size=bs, shuffle=True)
    te = DataLoader(DS([texts[i] for i in te_idx], y[te_idx].astype(int), tok, maxlen),
                    batch_size=bs)
    # class weighting: positives are 8-28% depending on the set, and an
    # unweighted loss simply predicts the majority everywhere.
    pos = max(float(y[tr_idx].mean()), 1e-6)
    w = torch.tensor([1.0, (1 - pos) / pos], dtype=torch.float, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    lossf = torch.nn.CrossEntropyLoss(weight=w)
    model.train()
    for _ in range(epochs):
        for batch in tr:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            out = model(**batch)
            loss = lossf(out.logits, labels)
            loss.backward(); opt.step(); opt.zero_grad()
    model.eval()
    scores = []
    with torch.no_grad():
        for batch in te:
            batch = {k: v.to(device) for k, v in batch.items()}
            batch.pop("labels")
            logits = model(**batch).logits
            scores.append(torch.softmax(logits, -1)[:, 1].float().cpu().numpy())
    del model
    return np.concatenate(scores)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="experiments/bbh_labels.jsonl")
    ap.add_argument("--items", default="experiments/bbh_items.jsonl")
    ap.add_argument("--group", default="subject")
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--loto", type=int, default=8, help="how many held-out-group folds")
    ap.add_argument("--out", default="results/reports/bert-router")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    recs = [json.loads(l) for l in (ROOT / args.labels).read_text().splitlines() if l.strip()]
    items = {json.loads(l)["id"]: json.loads(l)
             for l in (ROOT / args.items).read_text().splitlines() if l.strip()}
    recs = [r for r in recs if r["id"] in items]
    texts = [r["prompt"] for r in recs]
    g = np.array([str(items[r["id"]].get(args.group)) for r in recs])
    y = np.isin([r["tier_needed"] for r in recs], [1, 2]).astype(float)
    L = np.array([float(len(t)) for t in texts])
    print(f"{args.model} on {device} | n={len(recs)} groups={len(set(g))} "
          f"positives={int(y.sum())} ({y.mean():.1%})")

    rng = np.random.default_rng(20260921)
    perm = rng.permutation(len(recs)); cut = int(len(recs) * 0.70)
    tr, te = perm[:cut], perm[cut:]
    t0 = time.perf_counter()
    s = train_eval(texts, y, tr, te, args.model, args.epochs, args.lr,
                   args.batch_size, args.max_len, device)
    lo, hi = auc_ci(y[te], s)
    rand = {"auc": round(auc(y[te], s), 3), "ci95": [round(lo, 3), round(hi, 3)]}
    print(f"\nrandom split      AUC {rand['auc']:.3f}  [{lo:.3f}, {hi:.3f}]  "
          f"({time.perf_counter()-t0:.0f}s)")

    groups = sorted(set(g), key=lambda t: -int((g == t).sum()))[: args.loto]
    folds = []
    for grp in groups:
        m = g == grp
        if y[m].sum() < 5 or (1 - y[m]).sum() < 5:
            continue
        sc = train_eval(texts, y, np.where(~m)[0], np.where(m)[0], args.model,
                        args.epochs, args.lr, args.batch_size, args.max_len, device)
        a, l = auc(y[m], sc), auc(y[m], L[m])
        folds.append({"group": grp, "n": int(m.sum()), "bert": round(a, 3),
                      "length": round(l, 3)})
        print(f"  LOTO {grp:42} bert {a:.3f}   length {l:.3f}")

    mean_b = float(np.mean([f["bert"] for f in folds])) if folds else float("nan")
    mean_l = float(np.mean([f["length"] for f in folds])) if folds else float("nan")
    print(f"\nleave-one-group-out over {len(folds)} groups:")
    print(f"  bert   mean {mean_b:.3f}  (sd {np.std([f['bert'] for f in folds]):.3f})")
    print(f"  length mean {mean_l:.3f}")

    out = ROOT / f"{args.out}-{pathlib.Path(args.labels).stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": args.model, "labels": args.labels,
                               "random_split": rand, "loto": folds,
                               "loto_mean_bert": round(mean_b, 3),
                               "loto_mean_length": round(mean_l, 3)}, indent=2) + "\n")
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
