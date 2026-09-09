#!/usr/bin/env python3
"""
After 25k/50k budget runs finish, inject PRIMARY E2 f=0.5 as the 100k reference
(same protocol; different hashed seed stream — disclosed), then optionally continue 200k only.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ALGOS = ["MA", "ACMO-Diet", "BMOEA", "NSGA2", "RandomRepair", "Random"]


def inject_e2_100k(budget_df: pd.DataFrame) -> pd.DataFrame:
    e2 = pd.read_csv(ROOT / "results/raw/e2_sparsity_full.csv")
    e2 = e2[(e2["sparsity"] == 0.5) & (e2["scale"] == 10000) & (e2["algorithm"].isin(ALGOS))].copy()
    e2["max_evaluations"] = 100000
    e2["result_class"] = "BUDGET_SENSITIVITY_FROM_PRIMARY_E2"
    e2["experiment"] = "e_budget"
    # drop any existing 100k rows
    out = budget_df[budget_df["max_evaluations"] != 100000].copy()
    # align columns
    for c in out.columns:
        if c not in e2.columns:
            e2[c] = pd.NA
    e2 = e2[out.columns]
    return pd.concat([out, e2], ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partial", default="results/raw/e_budget_f05.csv.partial.csv")
    ap.add_argument("--out", default="results/raw/e_budget_f05.csv")
    ap.add_argument("--require-50k", action="store_true", default=True)
    args = ap.parse_args()
    partial = ROOT / args.partial
    if not partial.exists():
        raise SystemExit(f"missing {partial}")
    df = pd.read_csv(partial)
    n50 = int((df["max_evaluations"] == 50000).sum())
    n25 = int((df["max_evaluations"] == 25000).sum())
    print(f"loaded {len(df)} rows; n25={n25} n50={n50}")
    if args.require_50k and n50 < 720:
        raise SystemExit(f"50k incomplete ({n50}/720); not injecting yet")
    merged = inject_e2_100k(df)
    # drop incomplete 100k/200k live runs if any
    out_path = ROOT / args.out
    merged.to_csv(out_path, index=False)
    print(merged.groupby("max_evaluations").size())
    print("wrote", out_path)
    print("NOTE: 100k rows reused from PRIMARY E2 (seed stream differs; protocol matched).")


if __name__ == "__main__":
    main()
