#!/usr/bin/env python3
"""Render the paper's figures from committed results. No inference, no network.

Every number plotted is read from results/reports/*.json or recomputed from
cached embeddings, so a figure cannot drift from the analysis it illustrates.

Palette: the first three slots of a validated categorical order (blue, orange,
aqua), which pass the colour-vision-deficiency and normal-vision separation
checks for all pairs. Aqua sits below 3:1 contrast on the page, so its values
are labelled on the figure or tabulated in the text.

    .venv/bin/python paper/figures.py
"""
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
OUT = ROOT / "paper" / "figures"
REP = ROOT / "results" / "reports"

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
REF = "#9a9994"          # reference lines: chance, ceiling

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 9,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "legend.frameon": False, "figure.dpi": 200,
})


def load(name):
    return json.loads((REP / name).read_text())


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote paper/figures/{name}")


def dot(ax, x, y, color, **kw):
    """8px marker with a surface ring, so overlapping marks stay separable."""
    ax.plot(x, y, "o", ms=7.5, color=color, mec=SURFACE, mew=1.6, zorder=3, **kw)


# ---------------------------------------------------------------- figure 2
def fig_threshold_sweep():
    from threshold_sweep import load_embeddings, pgr_curve, scores, cost_per_1k

    rows = [json.loads(l) for l in
            (ROOT / "experiments/routellm_labels.jsonl").read_text().splitlines() if l.strip()]
    art = json.loads((ROOT / "router/artifacts/router_weights.json").read_text())
    X = load_embeddings([r["prompt"] for r in rows], 256)
    y = np.array([r["label"] for r in rows], float)
    cut = int(len(rows) * 0.85)
    Xte, yte = X[cut:], y[cut:]
    texts = [r["prompt"] for r in rows[cut:]]
    cands = [
        ("learned router", scores(Xte, np.array(art["weights"]), float(art["bias"])), BLUE),
        ("prompt length", np.array([float(len(t)) for t in texts]), ORANGE),
        ("random", np.random.default_rng(20260921).random(len(yte)), AQUA),
    ]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.1))
    for name, s, c in cands:
        r, p = pgr_curve(yte, s)
        a1.plot(r * 100, p * 100, color=c, lw=1.8, label=name)
        cost = [cost_per_1k(v, "claude-opus-4-5", "claude-haiku-4-5", 140, 280) for v in r]
        a2.plot(cost, p * 100, color=c, lw=1.8)
    for a in (a1, a2):
        a.axhline(80, color=REF, lw=0.9, ls=(0, (4, 3)), zorder=1)
        a.set_ylim(0, 100)
        a.set_ylabel("quality gap recovered (%)")
    a1.text(1, 82, "80% of the gap", color=INK2, fontsize=7.5)
    a1.set_xlim(0, 100)
    a1.set_xlabel("requests sent to the strong model (%)")
    a2.set_xlabel("cost per 1,000 requests, Haiku → Opus (USD)")
    full = cost_per_1k(1.0, "claude-opus-4-5", "claude-haiku-4-5", 140, 280)
    a2.axvline(full, color=REF, lw=0.9, ls=(0, (4, 3)))
    a2.text(full - 0.1, 4, f"always Opus\n${full:.2f}", ha="right", color=INK2, fontsize=7.5)
    a1.legend(loc="lower right", fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig2-threshold-sweep.png")


# ---------------------------------------------------------------- figure 3
def fig_what_routers_learn():
    sets = [("BBH\n27 tasks", "confounds-bbh_labels.json"),
            ("Mixed\n5 benchmarks", "confounds-cascade_labels.json"),
            ("MATH\n7 subjects", "confounds-math_labels.json")]
    ceiling = load("label-stability.json")["ceiling_auc"]
    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    for i, (label, f) in enumerate(sets):
        d = load(f)
        yv = len(sets) - 1 - i
        real, shuf, logo = d["auc_real"], d["auc_within_group_shuffle"], d["loo_mean_router"]
        ax.plot([logo, max(real, shuf)], [yv, yv], color=GRID, lw=2.2, zorder=1)
        dot(ax, real, yv, BLUE)
        dot(ax, shuf, yv - 0.0, ORANGE)
        dot(ax, logo, yv, AQUA)
        ax.text(logo - 0.008, yv, f"{logo:.3f}", ha="right", va="center", fontsize=7.5)
        ax.text(real + 0.008, yv, f"{real:.3f}", ha="left", va="center", fontsize=7.5)
        if i == 0:
            ax.text(real, yv - 0.32, f"shuffling labels within task costs {real - shuf:.3f}",
                    ha="right", va="top", fontsize=7.5, color=INK2)
    ax.axvline(0.5, color=REF, lw=0.9, ls=(0, (4, 3)), zorder=0)
    ax.axvline(ceiling, color=REF, lw=0.9, ls=(0, (1, 2)), zorder=0)
    ax.text(0.502, len(sets) - 0.45, "chance", color=INK2, fontsize=7.5)
    ax.text(ceiling - 0.004, len(sets) - 0.45, f"label ceiling {ceiling:.3f}",
            color=INK2, fontsize=7.5, ha="right")
    ax.set_yticks(range(len(sets)))
    ax.set_yticklabels([s for s, _ in reversed(sets)])
    ax.set_ylim(-0.8, len(sets) - 0.3)
    ax.set_xlim(0.45, 0.96)
    ax.set_xlabel("AUC, linear probe on Titan embeddings")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    handles = [plt.Line2D([], [], marker="o", ls="", ms=7, color=c, mec=SURFACE)
               for c in (BLUE, ORANGE, AQUA)]
    ax.legend(handles, ["in-distribution", "within-group shuffle", "leave-one-group-out"],
              loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=3, fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig3-what-routers-learn.png")


# ---------------------------------------------------------------- figure 4
def fig_mtbench():
    d = load("mtbench-diagnostic.json")
    rates = sorted(d["rate_by_category"].items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    ys = np.arange(len(rates))
    ax.barh(ys, [v * 100 for _, v in rates], height=0.62, color=BLUE, edgecolor=SURFACE,
            linewidth=1.5, zorder=2)
    for yv, (_, v) in zip(ys, rates):
        ax.text(v * 100 + 1.2, yv, f"{v:.0%}", va="center", fontsize=7.5, color=INK)
    overall = d["strong_needed_rate"] * 100
    ax.axvline(overall, color=REF, lw=0.9, ls=(0, (4, 3)), zorder=3)
    ax.text(overall + 1, -0.62, f"all categories {overall:.0f}%",
            color=INK2, fontsize=7.5)
    ax.set_yticks(ys)
    ax.set_yticklabels([k for k, _ in rates])
    ax.set_xlim(0, 75)
    ax.set_ylim(-0.9, len(rates) - 0.4)
    ax.set_xlabel("items where GPT-4 outscored Mixtral (%)")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_title(f"Category identity alone: AUC {d['category_oracle_auc']:.3f} "
                 f"[{d['category_oracle_ci95'][0]:.3f}, {d['category_oracle_ci95'][1]:.3f}]",
                 fontsize=8.5, color=INK, loc="left", pad=8)
    fig.tight_layout()
    save(fig, "fig4-mtbench-categories.png")


# ---------------------------------------------------------------- figure 5
def fig_deferral():
    d = load("deferral-math_labels.json")
    rows = sorted(d["loto_by_group"], key=lambda r: r["deferral"])
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    for yv, r in enumerate(rows):
        ax.plot([r["prompt_only"], r["deferral"]], [yv, yv], color=GRID, lw=2.2, zorder=1)
        dot(ax, r["prompt_only"], yv, ORANGE)
        dot(ax, r["deferral"], yv, BLUE)
    top = rows[-1]
    ax.text(top["deferral"], len(rows) - 1 + 0.35, "deferral", ha="center", fontsize=7.5)
    ax.text(top["prompt_only"], len(rows) - 1 + 0.35, "prompt-only", ha="center", fontsize=7.5)
    ax.axvline(0.5, color=REF, lw=0.9, ls=(0, (4, 3)), zorder=0)
    ax.text(0.503, -0.75, "chance", color=INK2, fontsize=7.5)
    m = d["loto_mean"]
    ax.set_title(f"Held-out subject, mean AUC: deferral {m['deferral']:.3f}, "
                 f"prompt-only {m['prompt_only']:.3f}", fontsize=8.5, loc="left", pad=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["group"].replace("_", " ") for r in rows])
    ax.set_xlim(0.45, 0.9)
    ax.set_ylim(-0.9, len(rows) - 0.3)
    ax.set_xlabel("AUC on the held-out MATH subject")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    handles = [plt.Line2D([], [], marker="o", ls="", ms=7, color=c, mec=SURFACE)
               for c in (BLUE, ORANGE)]
    ax.legend(handles, ["deferral (Haiku's own response)", "prompt-only router"],
              loc="lower right", fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig5-deferral-math.png")


if __name__ == "__main__":
    fig_threshold_sweep()
    fig_what_routers_learn()
    fig_mtbench()
    fig_deferral()
