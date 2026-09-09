#!/usr/bin/env python3
"""Analyze E0 mechanism-coverage check: merge PRIMARY E2 f=0.5 with new MOEAD rows."""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "e0_analysis"
OUT.mkdir(parents=True, exist_ok=True)

ALG_ORDER = [
    "MA", "ACMO-Diet", "BMOEA", "NSGA2", "MOEAD", "SPEA2",
    "RVEA", "CTAEA", "RandomRepair", "Greedy", "Random",
]


def seed_fhr(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["algorithm", "rep"], as_index=False)["FHR"].mean()
    return g.groupby("algorithm")["FHR"].agg(["mean", "std", "count"]).reset_index()


def main():
    e2 = pd.read_csv(ROOT / "results" / "raw" / "e2_sparsity_full.csv")
    hard = e2[e2["sparsity"] == 0.5].copy()
    moead_path = ROOT / "results" / "raw" / "e0_moead_f05.csv"
    if not moead_path.exists():
        # resume partial
        partial = Path(str(moead_path) + ".partial.csv")
        if partial.exists():
            moead_path = partial
        else:
            print("MOEAD results not ready:", moead_path)
            return
    mo = pd.read_csv(moead_path)
    mo = mo[mo["sparsity"] == 0.5].copy() if "sparsity" in mo.columns else mo
    # drop any accidental MOEAD from e2 if present
    hard = hard[hard["algorithm"] != "MOEAD"]
    df = pd.concat([hard, mo], ignore_index=True)
    df.to_csv(OUT / "e0_merged_f05.csv", index=False)

    summ = seed_fhr(df)
    summ = summ.set_index("algorithm").reindex([a for a in ALG_ORDER if a in set(summ["algorithm"])]).reset_index()
    summ.to_csv(OUT / "e0_fhr_seed_summary.csv", index=False)
    print(summ.to_string(index=False))

    # FFG among hits
    rows = []
    for a, g in df.groupby("algorithm"):
        hits = g[g["FHR"] >= 0.5]["FFG_evals"]
        rows.append({
            "algorithm": a,
            "n_success": int(len(hits)),
            "n_total": int(len(g)),
            "FHR_runmean": float(g["FHR"].mean()),
            "FFG_median": float(hits.median()) if len(hits) else np.nan,
            "FFG_iqr": float(hits.quantile(0.75) - hits.quantile(0.25)) if len(hits) else np.nan,
            "n_evals_median": float(g["n_evals"].median()),
        })
    ffg = pd.DataFrame(rows).sort_values("FHR_runmean", ascending=False)
    ffg.to_csv(OUT / "e0_ffg_summary.csv", index=False)
    print("\nFFG:\n", ffg.to_string(index=False))

    # Friedman on target-level mean FHR
    tgt = df.groupby(["target_id", "algorithm"], as_index=False)["FHR"].mean()
    piv = tgt.pivot(index="target_id", columns="algorithm", values="FHR")
    cols = [a for a in ALG_ORDER if a in piv.columns]
    piv = piv[cols]
    try:
        stat, p = stats.friedmanchisquare(*[piv[c].to_numpy() for c in cols])
        print(f"\nFriedman chi2={stat:.3f} p={p:.3e} algos={len(cols)} targets={len(piv)}")
    except Exception as e:
        print("Friedman failed", e)
        p = np.nan

    # Wilcoxon vs MA
    wil = []
    if "MA" in piv.columns:
        for a in cols:
            if a == "MA":
                continue
            x = piv["MA"].to_numpy()
            y = piv[a].to_numpy()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    w = stats.wilcoxon(x, y, zero_method="wilcox")
                    pv = float(w.pvalue)
                except Exception:
                    pv = np.nan
            # effect size: paired mean difference
            d = float(np.nanmean(x - y))
            wil.append({"vs": a, "mean_diff_MA_minus_other": d, "wilcoxon_p": pv})
    wil_df = pd.DataFrame(wil)
    wil_df.to_csv(OUT / "e0_wilcoxon_vs_MA.csv", index=False)
    print("\nvs MA:\n", wil_df.to_string(index=False))

    # ranks
    ranks = piv.rank(axis=1, ascending=False, method="average").mean().sort_values()
    ranks.to_csv(OUT / "e0_mean_ranks.csv", header=["mean_rank"])
    print("\nMean ranks (1=best):\n", ranks)


if __name__ == "__main__":
    main()
