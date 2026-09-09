#!/usr/bin/env python3
"""Combine 25k/50k/E2-100k budget file with 200k-only run into final curve CSV + figure."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    base = pd.read_csv(ROOT / "results/raw/e_budget_f05.csv")
    p200 = ROOT / "results/raw/e_budget_200k.csv"
    if p200.exists():
        b200 = pd.read_csv(p200)
        # align columns
        for c in base.columns:
            if c not in b200.columns:
                b200[c] = pd.NA
        b200 = b200[base.columns]
        out = pd.concat([base[base.max_evaluations != 200000], b200], ignore_index=True)
    else:
        out = base
    out_path = ROOT / "results/raw/e_budget_f05_final.csv"
    out.to_csv(out_path, index=False)
    print(out.groupby(["max_evaluations", "algorithm"])["FHR"].mean().unstack().round(3))
    print("wrote", out_path)
    # also copy to canonical name used by analyze_budget
    out.to_csv(ROOT / "results/raw/e_budget_f05.csv", index=False)


if __name__ == "__main__":
    main()
