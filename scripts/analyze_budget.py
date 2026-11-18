#!/usr/bin/env python3
"""Budget sensitivity plots/tables from e_budget_f05.csv."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    raw = ROOT / "results/raw/e_budget_f05.csv"
    if not raw.exists():
        raise SystemExit(f"missing {raw}")
    df = pd.read_csv(raw)
    g = df.groupby(["max_evaluations", "algorithm"], as_index=False).agg(
        mean_FHR=("FHR", "mean"),
        std_FHR=("FHR", "std"),
        n=("FHR", "count"),
    )
    g["ci95"] = 1.96 * g["std_FHR"] / g["n"].clip(lower=1) ** 0.5
    tab = ROOT / "results/tables"
    tab.mkdir(exist_ok=True)
    g.to_csv(tab / "e_budget_fhr_by_budget_algo.csv", index=False)

    ok = df[df["FHR"] > 0].copy()
    ffg_col = "FFG_evals" if "FFG_evals" in df.columns else (
        "ffg_evals" if "ffg_evals" in df.columns else None
    )
    if ffg_col:
        ffg = ok.groupby(["max_evaluations", "algorithm"], as_index=False)[ffg_col].median()
        ffg.to_csv(tab / "e_budget_ffg_median_success.csv", index=False)

    # Larger canvas + compact fonts so column-width scaling matches other figures.
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
        }
    )
    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    preferred = ["MA", "NSGA2", "RandomRepair", "Random"]
    algs = [a for a in preferred if a in set(g["algorithm"])]
    algs += [a for a in sorted(g["algorithm"].unique()) if a not in algs]
    for alg in algs:
        ag = g[g["algorithm"] == alg].sort_values("max_evaluations")
        ax.errorbar(
            ag["max_evaluations"],
            ag["mean_FHR"],
            yerr=ag["ci95"],
            marker="o",
            label=alg,
            lw=1.1,
            ms=3.5,
            capsize=2,
            elinewidth=0.8,
        )
    ax.set_xlabel("Evaluation budget")
    ax.set_ylabel("Mean FHR")
    ax.set_title(r"Budget sensitivity (NHANES-10K, $f=0.5$)")
    ax.set_xticks([25000, 50000, 100000, 200000])
    ax.set_xticklabels(["25k", "50k", "100k", "200k"])
    ax.set_ylim(0.0, 0.36)
    ax.set_xlim(10000, 215000)
    ax.grid(True, ls=":", alpha=0.4)

    handles, labels = ax.get_legend_handles_labels()
    # Outside below the axes: cannot overlap any curve/CI.
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=3,
        frameon=False,
        columnspacing=1.0,
        handlelength=1.6,
        handletextpad=0.35,
        borderaxespad=0.0,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    out = ROOT / "results/figures"
    out.mkdir(exist_ok=True)
    for name in ("fig_budget_vs_fhr.pdf", "fig_budget_vs_fhr.png"):
        fig.savefig(
            out / name,
            dpi=220 if name.endswith("png") else None,
            bbox_inches="tight",
            pad_inches=0.03,
        )
        figs = ROOT / "figs"
        figs.mkdir(exist_ok=True)
        fig.savefig(
            figs / name.replace("fig_", ""),
            dpi=220 if name.endswith("png") else None,
            bbox_inches="tight",
            pad_inches=0.03,
        )
    print(g.to_string(index=False))
    print("wrote budget figures")


if __name__ == "__main__":
    main()
