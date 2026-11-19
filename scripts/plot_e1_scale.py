#!/usr/bin/env python3
"""E1 figure: seed-level mean FHR of each solver against survey-derived pool size."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables"
OUT_DIRS = [ROOT / "results" / "figures", ROOT / "figs"]

ALG_ORDER = ["MA", "NSGA2", "SPEA2", "Greedy", "RandomRepair", "Random"]
SHORT = {"MA": "MA", "NSGA2": "NSGA-II", "SPEA2": "SPEA2", "Greedy": "Greedy",
         "RandomRepair": "RandomRepair", "Random": "Random"}
COLORS = {"MA": "#1b9e77", "NSGA2": "#d95f02", "SPEA2": "#7570b3", "Greedy": "#e6ab02",
          "RandomRepair": "#a6761d", "Random": "#666666"}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "axes.linewidth": 0.7,
})

s = pd.read_csv(TAB / "e1_8m_fhr_by_scale_algo.csv")
scales = sorted(s["scale"].unique())
labels = [r"$\approx$210K" if sc >= 200000 else f"{int(sc / 1000)}K" for sc in scales]

fig, ax = plt.subplots(figsize=(3.5, 2.4))
for alg in ALG_ORDER:
    sub = s[s["algorithm"] == alg].set_index("scale")
    if sub.empty:
        continue
    ax.plot(range(len(scales)), [sub.loc[sc, "mean"] for sc in scales], marker="o", ms=3.5,
            lw=1.1, color=COLORS[alg], label=SHORT[alg])
ax.set_xticks(range(len(scales)))
ax.set_xticklabels(labels)
ax.set_xlabel("Survey-derived pool size $N$")
ax.set_ylabel("Mean FHR")
ax.set_ylim(-0.05, 1.05)
ax.grid(True, ls=":", alpha=0.4)
ax.legend(frameon=False, ncol=2, loc="lower left", columnspacing=0.8, handlelength=1.4)
fig.tight_layout(pad=0.3)
for d in OUT_DIRS:
    d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / "fig_e1_full_fhr_scale.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(d / "fig_e1_full_fhr_scale.png", dpi=220, bbox_inches="tight", pad_inches=0.02)
print(s.pivot(index="scale", columns="algorithm", values="mean").round(3))
