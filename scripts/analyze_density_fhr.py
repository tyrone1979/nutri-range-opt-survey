#!/usr/bin/env python3
"""P0-1: empirical feasible density / violation vs algorithmic FHR (E2)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    dens = pd.read_csv(ROOT / "results" / "target_analysis" / "feasible_density.csv")
    e2 = pd.read_csv(ROOT / "results" / "raw" / "e2_sparsity_full.csv")
    submission_algorithms = {
        "MA", "NSGA2", "SPEA2", "Greedy", "RandomRepair", "Random"
    }
    e2 = e2[e2["algorithm"].isin(submission_algorithms)].copy()
    # target-level mean FHR across algos and seeds
    fhr = e2.groupby(["sparsity", "target_id"], as_index=False)["FHR"].mean()
    fhr = fhr.rename(columns={"sparsity": "sparsity_factor", "FHR": "mean_FHR"})
    m = dens.merge(fhr, on=["sparsity_factor", "target_id"], how="inner")
    # censored density for zeros: < 1/n_samples
    m["p_hat"] = m["feasible_density"].astype(float)
    m["p_plot"] = np.where(m["p_hat"] > 0, m["p_hat"], 1.0 / m["n_samples"].clip(lower=1))
    m["neglog10_p"] = -np.log10(m["p_plot"])
    m["censored"] = m["p_hat"] <= 0
    m.to_csv(ROOT / "results" / "tables" / "density_vs_fhr_by_target.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    colors = {0.5: "#d73027", 1.0: "#fc8d59", 1.5: "#91bfdb"}
    for f, g in m.groupby("sparsity_factor"):
        axes[0].scatter(
            g["neglog10_p"], g["mean_FHR"],
            c=colors.get(float(f), "gray"),
            label=rf"$f={f}$",
            alpha=0.85, s=36,
            marker="x" if False else "o",
        )
        # mark censored
        cens = g[g["censored"]]
        if len(cens):
            axes[0].scatter(
                cens["neglog10_p"], cens["mean_FHR"],
                facecolors="none", edgecolors=colors.get(float(f), "gray"),
                s=80, linewidths=1.2,
            )
    axes[0].set_xlabel(r"$-\log_{10}(\hat p_F)$ (open = censored at $1/n$)")
    axes[0].set_ylabel("Mean FHR (six methods)")
    axes[0].set_title("(a) Empirical sparsity vs FHR")
    axes[0].grid(True, ls=":", alpha=0.4)
    axes[0].legend(fontsize=8, frameon=False)

    for f, g in m.groupby("sparsity_factor"):
        axes[1].scatter(
            g["mean_violation"], g["mean_FHR"],
            c=colors.get(float(f), "gray"),
            label=rf"$f={f}$", alpha=0.85, s=36,
        )
    axes[1].set_xscale("log")
    axes[1].set_xlim(1e3, 5e4)
    axes[1].set_xticks([1e3, 2e3, 5e3, 1e4, 2e4, 5e4])
    axes[1].set_xticklabels(
        [r"$10^{3}$", r"$2{\times}10^{3}$", r"$5{\times}10^{3}$",
         r"$10^{4}$", r"$2{\times}10^{4}$", r"$5{\times}10^{4}$"],
        fontsize=7,
    )
    axes[1].minorticks_off()
    axes[1].set_xlabel("Mean nutrient violation (random samples)")
    axes[1].set_ylabel("Mean FHR")
    axes[1].set_title("(b) Violation proxy vs FHR")
    axes[1].grid(True, which="major", ls=":", alpha=0.4)
    axes[1].legend(fontsize=8, frameon=False)
    fig.suptitle(
        "Empirical feasible-region diagnostics (MC sampling; not exact volume)",
        fontsize=9, y=1.02,
    )
    fig.tight_layout()
    for name in ("fig_density_vs_fhr.pdf", "fig_density_vs_fhr.png"):
        fig.savefig(OUT / name, dpi=200 if name.endswith("png") else None, bbox_inches="tight")
        fig.savefig(ROOT / "figs" / name.replace("fig_", ""), dpi=200 if name.endswith("png") else None, bbox_inches="tight")
    plt.close(fig)

    # summary by factor
    summ = m.groupby("sparsity_factor").agg(
        mean_FHR=("mean_FHR", "mean"),
        mean_p=("p_hat", "mean"),
        mean_V=("mean_violation", "mean"),
        frac_censored=("censored", "mean"),
    ).reset_index()
    summ.to_csv(ROOT / "results" / "tables" / "density_fhr_summary_by_factor.csv", index=False)
    print(summ.to_string(index=False))
    print("wrote density vs FHR figure")


if __name__ == "__main__":
    (ROOT / "figs").mkdir(exist_ok=True)
    main()
