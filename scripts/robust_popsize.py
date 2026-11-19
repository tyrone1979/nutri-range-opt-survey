#!/usr/bin/env python3
"""Population-size robustness: pop 100 and 1000 against the primary pop 500.

Cells: E2 f=0.5, E2 f=1.0, E3 |J|=16 on the 10,000-item pool. Seeds are the
primary seeds, so each comparison changes only the population size.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results/raw"
EAS = ["MA", "NSGA2", "SPEA2"]


def load(path, key, level, pop):
    d = pd.read_csv(path)
    d = d[(d[key] == level) & d["algorithm"].isin(EAS + ["Random"])].copy()
    d["pop"] = pop
    return d


cells = [("E2 f=0.5", "e2_sparsity_full.csv", "robust_e2_pop{}.csv", "sparsity", 0.5),
         ("E2 f=1.0", "e2_sparsity_full.csv", "robust_e2_pop{}.csv", "sparsity", 1.0),
         ("E3 |J|=16", "swevo_e3_pymoo_corrected.csv", "robust_e3_pop{}.csv", "dim", 16)]


def kappa(pa, pr, n):
    pa = min(max(pa, 0.5 / n), 1 - 0.5 / n)
    pr = min(max(pr, 0.5 / n), 1 - 0.5 / n)
    return float(np.log(1 - pa) / np.log(1 - pr))


rows, tests = [], []
for name, prim, rob, key, level in cells:
    parts = [load(RAW / prim, key, level, 500)]
    for pop in (100, 1000):
        p = RAW / rob.format(pop)
        if p.exists():
            parts.append(load(p, key, level, pop))
    d = pd.concat(parts, ignore_index=True)
    d["hit"] = (d["FHR"] > 0).astype(int)
    rand = d[(d["algorithm"] == "Random") & (d["pop"] == 500)]
    pr, nr = rand["hit"].mean(), len(rand)
    for pop, g in d[d["algorithm"].isin(EAS)].groupby("pop"):
        seed_means = g.groupby(["algorithm", "rep"])["hit"].mean()
        tgt = g.groupby(["target_id", "algorithm"])["hit"].mean().unstack()
        stat = friedmanchisquare(*[tgt[a] for a in EAS]) if tgt.shape[0] > 1 else None
        for a in EAS:
            ga = g[g["algorithm"] == a]
            rows.append({"cell": name, "pop": pop, "algorithm": a,
                         "FHR": ga["hit"].mean(), "SD": seed_means[a].std(ddof=1),
                         "n": len(ga), "kappa": kappa(ga["hit"].mean(), pr, len(ga))})
        order = sorted(EAS, key=lambda a: -g[g["algorithm"] == a]["hit"].mean())
        tests.append({"cell": name, "pop": pop, "order": " > ".join(order),
                      "friedman_p": float(stat.pvalue) if stat else None,
                      "random_FHR_pop500": pr, "n_random": nr})
    base = d[(d["pop"] == 500) & d["algorithm"].isin(EAS)]
    for pop in (100, 1000):
        alt = d[(d["pop"] == pop) & d["algorithm"].isin(EAS)]
        if alt.empty:
            continue
        for a in EAS:
            x = base[base.algorithm == a].groupby("target_id")["hit"].mean()
            y = alt[alt.algorithm == a].groupby("target_id")["hit"].mean()
            x, y = x.align(y, join="inner")
            diff = (y - x)
            p = float(wilcoxon(x, y).pvalue) if (diff != 0).any() else 1.0
            tests.append({"cell": name, "pop": pop, "contrast": f"{a} pop{pop} vs 500",
                          "mean_diff": float(diff.mean()), "wilcoxon_p": p})

tab = pd.DataFrame(rows)
print(tab.pivot_table(index=["cell", "algorithm"], columns="pop", values="FHR").round(3))
print(tab.pivot_table(index=["cell", "algorithm"], columns="pop", values="kappa").round(2))
for t in tests:
    print(t)
(ROOT / "results/tables").mkdir(parents=True, exist_ok=True)
tab.to_csv(ROOT / "results/tables/robust_popsize.csv", index=False)
(ROOT / "results/stats/ROBUST_POPSIZE.json").write_text(json.dumps(tests, indent=2))
