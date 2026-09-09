#!/usr/bin/env python3
"""Analyze expanded exact calibration; classify A–D with explicit pool caveats."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    cal = pd.read_csv(ROOT / "results/raw/exact_multi_pool.csv")
    e2 = pd.read_csv(ROOT / "results/raw/e2_sparsity_full.csv")
    # PRIMARY algorithms on NHANES-10K, f=0.5 (NOT the same food sample as reduced-pool CBC)
    fhr = (
        e2[e2["sparsity"] == 0.5]
        .groupby("target_id", as_index=False)["FHR"]
        .mean()
        .rename(columns={"FHR": "algo_mean_FHR_10k_f05"})
    )
    out_rows = []
    for pool, g in cal.groupby("pool_size"):
        m = g.merge(fhr, on="target_id", how="left")
        for _, r in m.iterrows():
            lab = r["calibration_label"]
            af = float(r["algo_mean_FHR_10k_f05"])
            if lab == "proven_feasible" and af <= 0:
                abcd = "A_suggestive_feas_oracle_zero_algo_FHR"
            elif lab == "proven_feasible" and af > 0:
                abcd = "B_suggestive_feas_oracle_positive_algo_FHR"
            elif lab == "proven_infeasible":
                abcd = "C_proven_infeasible_reduced_pool"
            else:
                abcd = "D_unresolved"
            out_rows.append({
                "pool_size": int(pool),
                "target_id": r["target_id"],
                "dim": int(r["dim"]),
                "calibration_label": lab,
                "violation": r["violation"],
                "algo_mean_FHR_10k_f05": af,
                "abcd_label": abcd,
                "caveat": (
                    "CBC on reduced pool+candidate_limit; algo FHR on separate NHANES-10K sample. "
                    "A/B are suggestive only; C is certified under the reduced model; "
                    "never treat unresolved as infeasible."
                ),
            })
    out = pd.DataFrame(out_rows)
    tab_dir = ROOT / "results/tables"
    tab_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(tab_dir / "exact_multi_pool_abcd.csv", index=False)
    summ = out.groupby(["pool_size", "calibration_label"]).size().unstack(fill_value=0)
    summ.to_csv(tab_dir / "exact_multi_pool_summary.csv")
    abcd = out.groupby(["pool_size", "abcd_label"]).size().unstack(fill_value=0)
    abcd.to_csv(tab_dir / "exact_multi_pool_abcd_counts.csv")
    print(summ)
    print(abcd)
    print("Wrote", tab_dir / "exact_multi_pool_abcd.csv")


if __name__ == "__main__":
    main()
