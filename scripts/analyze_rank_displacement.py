#!/usr/bin/env python3
"""E4 supplement: mean rank + mean absolute rank displacement across regimes."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def ranks_from_fhr(df: pd.DataFrame, regime_cols, algo_col="algorithm", fhr_col="FHR"):
    """Per-regime Friedman-style mean ranks (1=best) over targets, then average."""
    # aggregate to target×algo mean FHR within each regime key
    gcols = list(regime_cols) + ["target_id", algo_col]
    agg = df.groupby(gcols, as_index=False)[fhr_col].mean()
    rows = []
    for key, g in agg.groupby(list(regime_cols)):
        # rank algorithms within each target (higher FHR better)
        rlist = []
        for tid, tg in g.groupby("target_id"):
            s = tg.set_index(algo_col)[fhr_col]
            # average rank for ties
            rk = s.rank(ascending=False, method="average")
            rlist.append(rk)
        mean_rank = pd.concat(rlist, axis=1).mean(axis=1)
        key_t = key if isinstance(key, tuple) else (key,)
        for alg, mr in mean_rank.items():
            rows.append({**{c: key_t[i] for i, c in enumerate(regime_cols)},
                         "algorithm": alg, "mean_rank": float(mr)})
    return pd.DataFrame(rows)


def main():
    e1 = pd.read_csv(ROOT / "results/raw/e1_scaling_full.csv")
    e2 = pd.read_csv(ROOT / "results/raw/e2_sparsity_full.csv")
    e3 = pd.read_csv(ROOT / "results/raw/e3_dimension_full.csv")
    # shared algorithms
    shared = sorted(set(e1.algorithm) & set(e2.algorithm) & set(e3.algorithm))
    e1 = e1[e1.algorithm.isin(shared)]
    e2 = e2[e2.algorithm.isin(shared)]
    e3 = e3[e3.algorithm.isin(shared)]

    r1 = ranks_from_fhr(e1, ["scale"])
    r1["regime"] = r1["scale"].astype(str)
    r2 = ranks_from_fhr(e2, ["sparsity"])
    r2["regime"] = "f=" + r2["sparsity"].astype(str)
    r3 = ranks_from_fhr(e3, ["dim"])
    r3["regime"] = "m=" + r3["dim"].astype(str)

    allr = pd.concat([
        r1[["regime", "algorithm", "mean_rank"]],
        r2[["regime", "algorithm", "mean_rank"]],
        r3[["regime", "algorithm", "mean_rank"]],
    ], ignore_index=True)

    wide = allr.pivot_table(index="algorithm", columns="regime", values="mean_rank")
    mean_rank = wide.mean(axis=1)
    # displacement vs each algorithm's own mean rank across regimes
    disp = wide.sub(mean_rank, axis=0).abs().mean(axis=1)
    out = pd.DataFrame({
        "algorithm": mean_rank.index,
        "mean_rank_across_regimes": mean_rank.values,
        "mean_abs_rank_displacement": disp.values,
    }).sort_values("mean_rank_across_regimes")
    tab = ROOT / "results/tables"
    tab.mkdir(parents=True, exist_ok=True)
    out.to_csv(tab / "e4_rank_displacement.csv", index=False)
    wide.to_csv(tab / "e4_ranks_by_regime.csv")
    print(out.to_string(index=False))
    print("Saved e4_rank_displacement.csv")


if __name__ == "__main__":
    main()
