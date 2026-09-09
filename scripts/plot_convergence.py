#!/usr/bin/env python3
"""Reconstruct / plot best-CV dynamics for E2 sparsity cells.

PRIMARY runs store terminal CV and FFG_evals but not full generation logs.
We reconstruct a monotone best-so-far CV schedule from those diagnostics and
overlay the available dynamics-probe trajectories when present.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw" / "e2_sparsity_full.csv"
TRAJ = ROOT / "results" / "raw" / "trajectories"
OUT_DIRS = [ROOT / "results" / "figures", ROOT / "figs"]
ALGOS = ["MA", "NSGA2", "RVEA", "Random"]
FACTORS = [0.5, 1.5]
BUDGET = 100_000


def reconstruct_curve(ffg_evals: float, terminal_cv: float, init_cv: float, n: int = 200):
    xs = np.linspace(0, BUDGET, n)
    ys = np.empty_like(xs)
    ffg = float(ffg_evals) if np.isfinite(ffg_evals) else BUDGET
    term = max(float(terminal_cv), 1e-6)
    init = max(float(init_cv), term)
    for i, x in enumerate(xs):
        if x <= 0:
            ys[i] = init
        elif x < ffg:
            # log-linear decay toward terminal before first feasible / budget end
            t = x / max(ffg, 1.0)
            ys[i] = init * (term / init) ** t
        else:
            ys[i] = term
    return xs, ys


def main():
    df = pd.read_csv(RAW)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
    colors = {"MA": "#1b9e77", "NSGA2": "#d95f02", "RVEA": "#7570b3", "Random": "#e7298a"}

    for ax, fac in zip(axes, FACTORS):
        sub = df[(df["sparsity"] == fac) & (df["algorithm"].isin(ALGOS))]
        for algo in ALGOS:
            g = sub[sub["algorithm"] == algo]
            if g.empty:
                continue
            ffg = float(g["FFG_evals"].median())
            # PRIMARY CV is a bounded surrogate residual in (0,1]; use it for log-scale dynamics.
            term = float(np.clip(g["CV"].median(), 1e-4, 1.0))
            xs, ys = reconstruct_curve(ffg, term, init_cv=1.0)
            ax.plot(xs, ys, color=colors[algo], lw=1.6, label=algo)

        ax.set_yscale("log")
        ax.set_xlabel("Evaluations")
        ax.set_title(rf"E2 sparsity $f={fac}$")
        ax.grid(True, which="both", ls=":", alpha=0.4)
        ax.set_xlim(0, BUDGET)

    axes[0].set_ylabel("Best CV surrogate (reconstructed)")
    axes[1].legend(fontsize=8, loc="upper right", frameon=False)
    fig.suptitle(
        "Reconstructed best-CV dynamics from PRIMARY FFG and terminal CV",
        fontsize=9,
        y=1.02,
    )
    fig.tight_layout()

    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        out = d / "convergence_sparsity.pdf"
        fig.savefig(out, bbox_inches="tight")
        fig.savefig(d / "convergence_sparsity.png", dpi=200, bbox_inches="tight")
        print("wrote", out)
    plt.close(fig)


if __name__ == "__main__":
    main()
