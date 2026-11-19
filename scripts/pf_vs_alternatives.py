#!/usr/bin/env python3
"""Does feasible frequency explain hit rates beyond simpler structural variables?

Food-derived cells (13): leave-one-axis-out MAE of the same two-parameter
complementary log-log curve, fitted on single candidate predictors.
Synthetic cells (E5): does log p_F add to the design factors m and f?
"""
from __future__ import annotations

import json
from math import comb, log10
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results/raw"
EAS = ["MA", "NSGA2", "SPEA2"]
B = 1e5


def log10_space(n, kmin=5, kmax=30):
    return log10(sum(comb(n, k) for k in range(kmin, kmax + 1)))


def rate(p, n):
    return min(max(p, 0.5 / n), 1 - 0.5 / n)


cells = pd.read_csv(ROOT / "results/tables/solution_space_collapse_cells.csv")
e1 = pd.read_csv(RAW / "swevo_e1_pymoo_corrected.csv")
e2 = pd.read_csv(RAW / "e2_sparsity_full.csv")
dim_e1 = e1.groupby("scale")["dim"].mean()
dim_e2 = e2.groupby("sparsity")["dim"].mean()

rows = []
for r in cells.itertuples():
    if r.axis == "E1":
        logx, J, f = log10_space(int(r.level)), dim_e1[int(r.level)], 1.0
    elif r.axis == "E2":
        logx, J, f = log10_space(10000), dim_e2[float(r.level)], float(r.level)
    else:
        logx, J, f = log10_space(10000), float(r.level), 1.0
    rows.append({"axis": r.axis, "level": r.level, "fhr_ea": r.fhr_ea, "n_ea": r.n_ea,
                 "log_pF": np.log10(r.p_F), "log_X": logx, "J": J, "f": f})
c = pd.DataFrame(rows)
c["y"] = [np.log(-np.log(1 - rate(p, n))) for p, n in zip(c["fhr_ea"], c["n_ea"])]


def loao(pred):
    out = {}
    for axis in ["E1", "E2", "E3"]:
        tr, te = c[c.axis != axis], c[c.axis == axis]
        if tr[pred].nunique() < 2:
            out[axis] = None
            continue
        b, a = np.polyfit(tr[pred], tr["y"], 1)
        p = 1 - np.exp(-np.exp(a + b * te[pred]))
        out[axis] = float(np.abs(p - te["fhr_ea"]).mean())
    return out


food = {}
for pred in ["log_pF", "log_X", "J", "f"]:
    rho, p = spearmanr(c[pred], c["fhr_ea"])
    food[pred] = {"spearman_rho": float(rho), "spearman_p": float(p), "loao_mae": loao(pred)}
print("Food-derived cells (13):")
print(c[["axis", "level", "fhr_ea", "log_pF", "log_X", "J", "f"]].round(3).to_string(index=False))
print(json.dumps(food, indent=2))

# ---------------- synthetic arm
e5 = pd.read_csv(RAW / "e5_synthetic_5mat.csv")
e5 = e5[e5.algorithm != "BMOEA"].copy()
e5["hit"] = (e5["FHR"] > 0).astype(int)
cr = []
for (N, m, f), g in e5.groupby(["scale", "dim", "sparsity"]):
    r = g[g.algorithm == "Random"]
    ea = g[g.algorithm.isin(EAS)]
    h, n = int(r.hit.sum()), len(r)
    cr.append({"N": N, "m": m, "f": f, "rand_hits": h,
               "log_pF": np.log10(-np.log(1 - rate(h / n, n)) / B),
               "ea_hits": int(ea.hit.sum()), "ea_n": len(ea), "ea_fhr": ea.hit.mean()})
s = pd.DataFrame(cr)
s["m"], s["f"] = s["m"].astype(str), s["f"].astype(str)
s["ea_miss"] = s["ea_n"] - s["ea_hits"]

base = smf.glm("ea_hits + ea_miss ~ C(m) * C(f)", data=s,
               family=sm.families.Binomial()).fit()
full = smf.glm("ea_hits + ea_miss ~ C(m) * C(f) + log_pF", data=s,
               family=sm.families.Binomial()).fit()
lr = 2 * (full.llf - base.llf)
from scipy.stats import chi2
syn = {
    "design_only_deviance": float(base.deviance),
    "design_plus_pF_deviance": float(full.deviance),
    "lr_chi2_1": float(lr), "lr_p": float(chi2.sf(lr, 1)),
    "pF_coef": float(full.params["log_pF"]), "pF_se": float(full.bse["log_pF"]),
}
# within-stratum rank agreement: demean within m x f
s["ea_res"] = s["ea_fhr"] - s.groupby(["m", "f"])["ea_fhr"].transform("mean")
s["pf_res"] = s["log_pF"] - s.groupby(["m", "f"])["log_pF"].transform("mean")
rho, p = spearmanr(s["pf_res"], s["ea_res"])
syn["within_stratum_spearman"] = float(rho)
syn["within_stratum_p"] = float(p)
print("\nSynthetic cells (18):")
print(s[["N", "m", "f", "rand_hits", "log_pF", "ea_fhr"]].round(3).to_string(index=False))
print(json.dumps(syn, indent=2))

(ROOT / "results/stats").mkdir(parents=True, exist_ok=True)
(ROOT / "results/stats/PF_VS_ALTERNATIVES.json").write_text(json.dumps({"food": food, "synthetic": syn}, indent=2))
