#!/usr/bin/env python3
"""Boxplots of per-seed FHR for E3 dimension-16."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw" / "e3_dimension_full.csv"
OUT_DIRS = [ROOT / "results" / "figures", ROOT / "figs"]
ALGOS = ["MA", "NSGA2", "RVEA", "CTAEA", "Random"]


def main():
    df = pd.read_csv(RAW)
    d16 = df[df["dim"] == 16]
    # seed-level FHR: mean over targets within each rep
    seed = d16.groupby(["algorithm", "rep"], as_index=False)["FHR"].mean()

    data, means = [], []
    labels = []
    for algo in ALGOS:
        vals = seed.loc[seed["algorithm"] == algo, "FHR"].to_numpy()
        if len(vals) == 0:
            continue
        data.append(vals)
        means.append(vals.mean())
        labels.append(algo)

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.55, showfliers=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#c6dbef")
        patch.set_edgecolor("#08519c")
    ax.scatter(range(1, len(means) + 1), means, marker="D", s=28, color="#a50f15", zorder=5, label="mean")
    ax.set_ylabel("FHR (per seed)")
    ax.set_xlabel("Algorithm")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("E3 dimension 16: FHR across 10 seeds")
    ax.grid(True, axis="y", ls=":", alpha=0.4)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    fig.tight_layout()

    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / "fhr_distribution_dim16.pdf", bbox_inches="tight")
        fig.savefig(d / "fhr_distribution_dim16.png", dpi=200, bbox_inches="tight")
        print("wrote", d / "fhr_distribution_dim16.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
