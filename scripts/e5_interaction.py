#!/usr/bin/env python3
"""E5 factorial analysis: N x m x f interactions on synthetic matrices.

Unit of replication is the generated matrix (instance). FHR is averaged over
solvers and seeds within each instance, then a three-way ANOVA (type II) is
fitted on instance-level means. A binomial GLM on run-level hits with solver
as an additional factor gives likelihood-ratio tests for the interaction terms.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parents[1]
src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/raw/e5_synthetic_5mat.csv"
out_dir = ROOT / "results/stats"
out_dir.mkdir(parents=True, exist_ok=True)

d = pd.read_csv(src)
d = d[d["algorithm"] != "BMOEA"].copy()
parts = d["target_id"].str.extract(r"N(?P<N>\d+)_m(?P<m>\d+)_f(?P<f>[\d.]+)_i(?P<inst>\d+)")
d = pd.concat([d, parts], axis=1)
d["hit"] = (d["FHR"] > 0).astype(int)
for c in ("N", "m", "f", "inst"):
    d[c] = d[c].astype(str)
print(f"rows={len(d)} instances per cell:")
print(d.groupby(["N", "m", "f"])["inst"].nunique().describe())

inst = d.groupby(["N", "m", "f", "inst"], as_index=False)["hit"].mean()
if inst.groupby(["N", "m", "f"]).size().min() >= 2:
    ols = smf.ols("hit ~ C(N) * C(m) * C(f)", data=inst).fit()
    aov = sm.stats.anova_lm(ols, typ=2)
    ss_res = aov.loc["Residual", "sum_sq"]
    aov["partial_eta2"] = aov["sum_sq"] / (aov["sum_sq"] + ss_res)
    aov.loc["Residual", "partial_eta2"] = np.nan
    print("\nInstance-level three-way ANOVA (type II):")
    print(aov.round(4))
else:
    aov = pd.DataFrame()
    print("\nOne matrix per cell: instance-level ANOVA skipped")

main = smf.glm("hit ~ C(algorithm) + C(N) + C(m) + C(f)", data=d,
               family=sm.families.Binomial()).fit()
two = smf.glm("hit ~ C(algorithm) + (C(N) + C(m) + C(f))**2", data=d,
              family=sm.families.Binomial()).fit()
three = smf.glm("hit ~ C(algorithm) + C(N) * C(m) * C(f)", data=d,
                family=sm.families.Binomial()).fit()
algo_x = smf.glm("hit ~ C(algorithm) * (C(N) + C(m) + C(f))", data=d,
                 family=sm.families.Binomial()).fit()


def lrt(small, big):
    stat = 2 * (big.llf - small.llf)
    df = big.df_model - small.df_model
    return {"chi2": float(stat), "df": int(df), "p": float(chi2.sf(stat, df))}


glm = {
    "two_way_vs_main": lrt(main, two),
    "three_way_vs_two_way": lrt(two, three),
    "solver_by_factor_vs_main": lrt(main, algo_x),
    "deviance_explained_main": float(1 - main.deviance / main.null_deviance),
    "deviance_explained_three_way": float(1 - three.deviance / three.null_deviance),
}
print("\nRun-level binomial GLM likelihood-ratio tests:")
print(json.dumps(glm, indent=2))

cell = d.groupby(["m", "f"])["hit"].mean().unstack()
print("\nMean FHR by m x f:")
print(cell.round(3))
cell_nm = d.groupby(["N", "m"])["hit"].mean().unstack()
print("\nMean FHR by N x m:")
print(cell_nm.round(3))
spread = inst.groupby(["N", "m", "f"])["hit"].agg(["mean", "std", "min", "max", "count"])
print("\nAcross-matrix spread per cell:")
print(spread.round(3))
by_alg = d.groupby(["algorithm", "m", "f"])["hit"].mean().unstack(["m", "f"])
print("\nFHR by solver x (m, f):")
print(by_alg.round(3))

res = {
    "source": str(src),
    "n_rows": int(len(d)),
    "anova": {k: {kk: (None if pd.isna(vv) else float(vv)) for kk, vv in row.items()}
              for k, row in aov.iterrows()},
    "glm": glm,
    "fhr_m_by_f": {f"m{m}": {f"f{f}": float(v) for f, v in r.items()} for m, r in cell.iterrows()},
    "fhr_N_by_m": {f"N{n}": {f"m{m}": float(v) for m, v in r.items()} for n, r in cell_nm.iterrows()},
    "matrix_spread": spread.reset_index().to_dict("records"),
}
(out_dir / "E5_INTERACTION.json").write_text(json.dumps(res, indent=2, default=str))
spread.to_csv(ROOT / "results/tables/e5_matrix_spread.csv")
print(f"\nSaved {out_dir / 'E5_INTERACTION.json'}")
