#!/usr/bin/env python3
"""Summarize and plot the corrected budget sensitivity panel (NHANES-10K, f=0.5)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results/raw/e_budget_f05_corrected.csv"
FIG = ROOT / "results/figures"
TAB = ROOT / "results/tables"
STAT = ROOT / "results/stats"

COLORS = {
    "MA": "#1b9e77",
    "NSGA2": "#d95f02",
    "RandomRepair": "#a6761d",
    "Random": "#666666",
}
LABELS = {
    "MA": "MA",
    "NSGA2": "NSGA-II",
    "RandomRepair": "RandomRepair",
    "Random": "Random",
}
ORDER = ["MA", "NSGA2", "RandomRepair", "Random"]


def main() -> None:
    df = pd.read_csv(RAW)
    df = df[df["algorithm"].isin(ORDER)].copy()
    # Seed-level mean across 12 targets, then mean and SD across 10 seeds (matches Table 2)
    seed_means = df.groupby(["max_evaluations", "algorithm", "rep"], as_index=False)["FHR"].mean()
    g = seed_means.groupby(["max_evaluations", "algorithm"], as_index=False).agg(
        mean_FHR=("FHR", "mean"),
        sd_FHR=("FHR", "std"),
        n_seeds=("FHR", "count"),
    )
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    STAT.mkdir(parents=True, exist_ok=True)
    g.to_csv(TAB / "e_budget_fhr_by_budget_algo.csv", index=False)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Nimbus Roman", "Times", "DejaVu Serif"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "axes.linewidth": 0.7,
    })
    fig, ax = plt.subplots(figsize=(4.4, 2.7))
    for alg in ORDER:
        ag = g[g["algorithm"] == alg].sort_values("max_evaluations")
        ax.errorbar(
            ag["max_evaluations"], ag["mean_FHR"], yerr=ag["sd_FHR"],
            marker="o", label=LABELS[alg], lw=1.2, ms=4, capsize=2.0,
            elinewidth=0.8, color=COLORS[alg],
        )
    ax.set_xlabel("Evaluation budget")
    ax.set_ylabel("Mean FHR (seed mean $\\pm$ SD)")
    ax.set_xticks([25000, 50000, 100000, 200000])
    ax.set_xticklabels(["25k", "50k", "100k", "200k"])
    ymax = max(0.65, float((g["mean_FHR"] + g["sd_FHR"]).max()) + 0.08)
    ax.set_ylim(-0.02, min(1.05, ymax))
    ax.grid(True, ls=":", alpha=0.45)
    ax.legend(loc="upper left", ncol=2, frameon=False, columnspacing=0.9, handlelength=1.4)
    fig.tight_layout(pad=0.3)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig_budget_vs_fhr.{ext}", dpi=220 if ext == "png" else None,
                    bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    summary = {}
    for b, sub in g.groupby("max_evaluations"):
        summary[int(b)] = {
            "mean4": float(sub["mean_FHR"].mean()),
            "per_algo": {r.algorithm: {"mean": round(float(r.mean_FHR), 4),
                                       "sd": round(float(r.sd_FHR), 4)}
                         for r in sub.itertuples()},
        }
    (STAT / "BUDGET_CORRECTED_SUMMARY.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
