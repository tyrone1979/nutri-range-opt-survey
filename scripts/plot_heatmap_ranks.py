#!/usr/bin/env python3
"""Heatmap of cross-regime mean Friedman ranks (E4)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RANKS = ROOT / "results" / "tables" / "e4_cross_regime_mean_ranks.csv"
OUT_DIRS = [ROOT / "results" / "figures", ROOT / "figs"]


def main():
    df = pd.read_csv(RANKS)
    # Exclude solvers missing E3 for a clean 8x3 shared panel when possible;
    # keep all rows but mask NaN.
    cols = ["E1_NHANES", "E2_sparsity", "E3_dimension"]
    mat = df[cols].to_numpy(dtype=float)
    algos = df["algorithm"].tolist()

    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    cmap = plt.get_cmap("RdYlBu_r")  # dark blue ~ rank1 best; red ~ worst
    vmax = np.nanmax(mat)
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=1.0, vmax=vmax)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["E1", "E2", "E3"])
    ax.set_yticks(range(len(algos)))
    ax.set_yticklabels(algos, fontsize=8)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isnan(v):
                ax.text(j, i, "---", ha="center", va="center", fontsize=7, color="white")
            else:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if v > (1 + vmax) / 2 else "black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean Friedman rank (1=best)", fontsize=8)
    ax.set_title("E4 rank concordance heatmap", fontsize=9)
    fig.tight_layout()

    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / "rank_concordance_heatmap.pdf", bbox_inches="tight")
        fig.savefig(d / "rank_concordance_heatmap.png", dpi=200, bbox_inches="tight")
        print("wrote", d / "rank_concordance_heatmap.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
