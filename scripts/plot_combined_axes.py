#!/usr/bin/env python3
"""Composite figures for the width axis (E2) and the dimensionality axis (E3)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"
TAB = ROOT / "results" / "tables"
OUT_DIRS = [ROOT / "results" / "figures", ROOT / "figs"]

ALG_ORDER = ["MA", "NSGA2", "SPEA2", "Greedy", "RandomRepair", "Random"]
SHORT = {"MA": "MA", "NSGA2": "NSGA-II", "SPEA2": "SPEA2", "Greedy": "Greedy",
         "RandomRepair": "RandomRepair", "Random": "Random"}
COLORS = {"MA": "#1b9e77", "NSGA2": "#d95f02", "SPEA2": "#7570b3", "Greedy": "#e6ab02",
          "RandomRepair": "#a6761d", "Random": "#666666"}
WIDTH_COLORS = {0.5: "#d73027", 1.0: "#fc8d59", 1.5: "#4575b4"}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def save(fig, stem):
    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.02)
        fig.savefig(d / f"{stem}.png", dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", stem)


def solver_lines(ax, table, xcol, xs):
    for alg in ALG_ORDER:
        sub = table[table["algorithm"] == alg]
        if sub.empty:
            continue
        ys = [float(sub.loc[np.isclose(sub[xcol], x), "mean"].iloc[0]) for x in xs]
        ax.plot(range(len(xs)), ys, marker="o", ms=3.2, lw=1.1, color=COLORS[alg], label=SHORT[alg])
    ax.set_xticks(range(len(xs)))
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Mean FHR")
    ax.grid(True, ls=":", alpha=0.4)


def width_figure():
    e2 = pd.read_csv(TAB / "e2_8m_fhr_by_sparsity_algo.csv")
    dens = pd.read_csv(TAB / "density_vs_fhr_by_target.csv")
    factors = [0.5, 1.0, 1.5]

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.35))
    ax = axes[0]
    solver_lines(ax, e2, "sparsity", factors)
    ax.set_xticklabels(["0.5", "1.0", "1.5"])
    ax.set_xlabel("Interval-width factor $f$")
    ax.set_title("(a) Hit rate by solver")
    ax.legend(frameon=False, ncol=2, loc="lower right", columnspacing=0.6, handlelength=1.2)

    ax = axes[1]
    for f, g in dens.groupby("sparsity_factor"):
        c = WIDTH_COLORS.get(float(f), "gray")
        hit = g[~g["censored"]]
        cen = g[g["censored"]]
        ax.scatter(hit["neglog10_p"], hit["mean_FHR"], color=c, s=18, alpha=0.85, label=rf"$f={f}$")
        ax.scatter(cen["neglog10_p"], cen["mean_FHR"], facecolors="none", edgecolors=c, s=26, linewidths=0.9)
    ax.set_xlabel(r"$-\log_{10}\hat p_F$ (2000 draws; open = censored)")
    ax.set_title(r"(b) Feasible frequency")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(frameon=False, loc="lower left", handlelength=1.0)

    ax = axes[2]
    for f, g in dens.groupby("sparsity_factor"):
        ax.scatter(g["mean_violation"], g["mean_FHR"], color=WIDTH_COLORS.get(float(f), "gray"),
                   s=18, alpha=0.85, label=rf"$f={f}$")
    ax.set_xscale("log")
    ax.set_xlim(1e3, 2e4)
    ax.set_xticks([1e3, 3e3, 1e4, 2e4])
    ax.set_xticklabels([r"$10^{3}$", r"$3{\times}10^{3}$", r"$10^{4}$", r"$2{\times}10^{4}$"])
    ax.minorticks_off()
    ax.set_xlabel("Mean violation of random subsets")
    ax.set_title("(c) Violation level")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, ls=":", alpha=0.4)
    for a in axes[1:]:
        a.set_ylabel("Mean FHR (target)")
    fig.tight_layout(pad=0.3, w_pad=0.8)
    save(fig, "fig_width_axis")


def e3_raw():
    df = pd.read_csv(RAW / "e3_dimension_full.csv")
    corrected = RAW / "swevo_e3_pymoo_corrected.csv"
    if corrected.exists():
        c = pd.read_csv(corrected)
        df = pd.concat([df[~df["algorithm"].isin(set(c["algorithm"]))], c], ignore_index=True)
    return df[df["algorithm"].isin(ALG_ORDER)]


def dimension_figure():
    e3 = pd.read_csv(TAB / "e3_8m_fhr_by_dim_algo.csv")
    dims = sorted(e3["dim"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    solver_lines(ax, e3, "dim", dims)
    ax.set_xticklabels([str(int(d)) for d in dims])
    ax.set_xlabel("Constrained coordinates $|J|$")
    ax.set_title("(a) Hit rate by solver")
    ax.legend(frameon=False, loc="lower left", handlelength=1.2)

    raw = e3_raw()
    d16 = raw[raw["dim"] == 16].groupby(["algorithm", "rep"], as_index=False)["FHR"].mean()
    algs = [a for a in ALG_ORDER if a in set(d16["algorithm"])]
    data = [d16.loc[d16["algorithm"] == a, "FHR"].to_numpy() for a in algs]
    ax = axes[1]
    bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor("#c6dbef")
        patch.set_edgecolor("#08519c")
    rng = np.random.default_rng(0)
    for i, v in enumerate(data, start=1):
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=7, color="#08306b", alpha=0.6, zorder=4)
    ax.scatter(range(1, len(data) + 1), [v.mean() for v in data], marker="D", s=16,
               color="#a50f15", zorder=5, label="mean")
    ax.set_xticks(range(1, len(algs) + 1))
    ax.set_xticklabels([SHORT[a] for a in algs])
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("FHR per seed")
    ax.set_title(r"(b) Seed-level spread at $|J|=16$")
    ax.grid(True, axis="y", ls=":", alpha=0.4)
    ax.legend(frameon=False, loc="lower left", handlelength=1.0)
    fig.tight_layout(pad=0.3, w_pad=1.0)
    save(fig, "fig_dimension_axis")
    print({SHORT[a]: round(float(v.mean()), 3) for a, v in zip(algs, data)})


if __name__ == "__main__":
    width_figure()
    dimension_figure()
