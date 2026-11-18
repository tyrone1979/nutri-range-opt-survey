#!/usr/bin/env python3
"""Boxplots of per-seed FHR for E3 dimension-16."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_CANDIDATES = [
    ROOT / "results" / "raw" / "swevo_e3_pymoo_corrected.csv",
    ROOT / "results" / "raw" / "e3_dimension_full.csv",
]
OUT_DIRS = [ROOT / "results" / "figures", ROOT / "figs"]
ALGOS = ["MA", "NSGA2", "SPEA2", "Random"]
SHORT = {
    "NSGA2": "NSGA-II",
}


def main():
    raw = next(p for p in RAW_CANDIDATES if p.exists())
    df = pd.read_csv(raw)
    d16 = df[df["dim"] == 16]
    seed = d16.groupby(["algorithm", "rep"], as_index=False)["FHR"].mean()

    data, means, labels = [], [], []
    for algo in ALGOS:
        vals = seed.loc[seed["algorithm"] == algo, "FHR"].to_numpy()
        if len(vals) == 0:
            continue
        data.append(vals)
        means.append(float(vals.mean()))
        labels.append(SHORT.get(algo, algo))

    # Match PRIMARY figures (budget / E4 heatmap): serif ~7 pt, angled x-ticks.
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Nimbus Roman", "Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 5.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6,
            "axes.linewidth": 0.7,
        }
    )

    fig, ax = plt.subplots(figsize=(3.5, 2.45))
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.55, showfliers=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#c6dbef")
        patch.set_edgecolor("#08519c")
    ax.scatter(
        range(1, len(means) + 1),
        means,
        marker="D",
        s=14,
        color="#a50f15",
        zorder=5,
        label="mean",
    )
    ax.set_ylabel("FHR (per seed)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("E3 dimension 16: FHR across 10 seeds")
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=35, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="x", length=2.5, pad=1)
    ax.grid(True, axis="y", ls=":", alpha=0.4)
    ax.legend(frameon=False, loc="lower left", handlelength=1.1, borderpad=0.15)
    fig.subplots_adjust(left=0.14, right=0.98, top=0.88, bottom=0.28)

    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / "fhr_distribution_dim16.pdf", bbox_inches="tight", pad_inches=0.02)
        fig.savefig(d / "fhr_distribution_dim16.png", dpi=220, bbox_inches="tight", pad_inches=0.02)
        print("wrote", d / "fhr_distribution_dim16.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
