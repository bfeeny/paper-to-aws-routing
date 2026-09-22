#!/usr/bin/env python3
"""Does the BBH router predict difficulty, or just recognize the task?

The mixed-benchmark result was confounded by corpus style. Restricting to BBH
removes that, and the in-distribution AUC rises to 0.833. This asks the same
question one level down, because BBH is 27 tasks whose escalation rates run
from 0% to 83% -- and a prompt's task is trivially identifiable from its
template.

Two tests:
  * a task-identity oracle, knowing only the training escalation rate per task
  * held-out *tasks*: train on 20, score 7 the model has never seen

    .venv/bin/python analysis/bbh_task_ablation.py
"""

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cascade_sweep import auc, auc_ci  # noqa: E402
from train_router import embed, fit_logistic, scores  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent


def main() -> int:
    recs = [json.loads(l) for l in (ROOT / "experiments/bbh_labels.jsonl").read_text().splitlines() if l.strip()]
    items = {json.loads(l)["id"]: json.loads(l)
             for l in (ROOT / "experiments/bbh_items.jsonl").read_text().splitlines() if l.strip()}
    task = np.array([items[r["id"]]["subject"] for r in recs if r["id"] in items])
    recs = [r for r in recs if r["id"] in items]
    y = np.isin([r["tier_needed"] for r in recs], [1, 2]).astype(float)
    X = embed([r["prompt"] for r in recs], "personal", "us-east-1", 256, workers=24)
    length = np.array([float(len(r["prompt"])) for r in recs])

    rng = np.random.default_rng(20260921)
    perm = rng.permutation(len(recs))
    cut = int(len(recs) * 0.70)
    tr, te = perm[:cut], perm[cut:]
    w, b = fit_logistic(X[tr], y[tr], l2=1.0)
    s = scores(X, w, b)

    rate = {t: float(y[tr][task[tr] == t].mean()) for t in sorted(set(task))}
    oracle = np.array([rate[t] for t in task])
    out = {"random_split": {
        "router_auc": round(auc(y[te], s[te]), 3),
        "task_identity_oracle_auc": round(auc(y[te], oracle[te]), 3),
        "oracle_ci95": [round(v, 3) for v in auc_ci(y[te], oracle[te])],
        "escalation_rate_by_task": {k: round(v, 3) for k, v in
                                    sorted(rate.items(), key=lambda kv: -kv[1])}}}

    # held-out tasks: the only split that asks whether difficulty generalizes
    ts = sorted(set(task))
    rng2 = np.random.default_rng(7)
    rng2.shuffle(ts)
    hold = set(ts[:7])
    m_te = np.isin(task, list(hold))
    w2, b2 = fit_logistic(X[~m_te], y[~m_te], l2=1.0)
    s2 = scores(X, w2, b2)
    out["held_out_tasks"] = {
        "tasks": sorted(hold), "n": int(m_te.sum()), "positives": int(y[m_te].sum()),
        "router_auc": round(auc(y[m_te], s2[m_te]), 3),
        "router_ci95": [round(v, 3) for v in auc_ci(y[m_te], s2[m_te])],
        "prompt_length_auc": round(auc(y[m_te], length[m_te]), 3),
        "prompt_length_ci95": [round(v, 3) for v in auc_ci(y[m_te], length[m_te])]}

    path = ROOT / "results/reports/bbh-task-ablation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    print(f"\nwrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
