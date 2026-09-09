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

    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    for alg, ag in g.groupby("algorithm"):
        ax.errorbar(
            ag["max_evaluations"],
            ag["mean_FHR"],
            yerr=ag["ci95"],
            marker="o",
            label=alg,
            lw=1.2,
            ms=4,
            capsize=2,
        )
    ax.set_xlabel("Evaluation budget")
    ax.set_ylabel("Mean FHR")
    ax.set_title(r"Budget sensitivity (NHANES-10K, $f=0.5$)")
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    out = ROOT / "results/figures"
    out.mkdir(exist_ok=True)
    for name in ("fig_budget_vs_fhr.pdf", "fig_budget_vs_fhr.png"):
        fig.savefig(out / name, dpi=200 if name.endswith("png") else None, bbox_inches="tight")
        figs = ROOT / "figs"
        figs.mkdir(exist_ok=True)
        fig.savefig(figs / name.replace("fig_", ""), dpi=200 if name.endswith("png") else None, bbox_inches="tight")
    print(g.to_string(index=False))
    print("wrote budget figures")


if __name__ == "__main__":
    main()
