#!/usr/bin/env python3
"""Target-space coverage: compare random / stratified / cluster representatives."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


NUTRIENTS = [
    "calorie", "protein", "carb", "sugar", "fiber", "saturated_fat", "cholesterol",
    "sodium", "potassium", "calcium", "vitamin_c", "vitamin_d", "iron", "folic_acid",
    "vitamin_b12", "phosphorus",
]


def parse_targets(path="data/target.csv"):
    df = pd.read_csv(path)
    rows = []
    for i in range(len(df)):
        td = eval(df.iloc[i, 0])
        feat = []
        for n in NUTRIENTS:
            if n in td:
                lo, hi = td[n]
                mid = 0.5 * (lo + hi)
                width = hi - lo
                present = 1.0
            else:
                mid = width = 0.0
                present = 0.0
            feat.extend([present, mid, width])
        rows.append(feat)
    X = np.array(rows, dtype=float)
    return X


def load_rep_indices(path, idx_col="original_idx"):
    df = pd.read_csv(path)
    if idx_col in df.columns:
        return df[idx_col].astype(int).tolist()
    return list(range(len(df)))


def main():
    os.makedirs("results/target_analysis", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    X = parse_targets()
    n = len(X)
    rng = np.random.default_rng(42)

    sets = {
        "all": np.arange(n),
        "random_12": rng.choice(n, 12, replace=False),
        "random_45": rng.choice(n, 45, replace=False),
        "cluster_12": load_rep_indices("data/representative_targets.csv"),
        "cluster_45": load_rep_indices("data/representative_targets_45.csv"),
        "cluster_100": load_rep_indices("data/representative_targets_100.csv"),
    }

    scaler = StandardScaler()
    Xs = scaler.fit_transform(np.nan_to_num(X))
    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(Xs)

    coverage_rows = []
    for name, idx in sets.items():
        idx = np.array(idx)
        idx = idx[(idx >= 0) & (idx < n)]
        sub = Xs[idx]
        # coverage proxy: mean nearest-neighbor distance from all points to subset
        # (subsample all for speed)
        probe = Xs[rng.choice(n, size=min(800, n), replace=False)]
        # distances
        dmin = []
        for p in probe:
            dmin.append(np.min(np.linalg.norm(sub - p, axis=1)))
        coverage_rows.append({
            "set": name,
            "k": len(idx),
            "mean_nn_dist": float(np.mean(dmin)),
            "median_nn_dist": float(np.median(dmin)),
            "pca_var_explained": float(pca.explained_variance_ratio_.sum()),
        })

    cov = pd.DataFrame(coverage_rows)
    cov.to_csv("results/target_coverage.csv", index=False)
    cov.to_csv("results/target_analysis/target_coverage.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(Z[:, 0], Z[:, 1], s=4, alpha=0.15, c="#888888", label="all targets")
    for name, color in [("cluster_12", "#c0392b"), ("random_12", "#2980b9")]:
        idx = np.array(sets[name])
        ax.scatter(Z[idx, 0], Z[idx, 1], s=40, c=color, label=name, edgecolors="k", linewidths=0.3)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Target-space coverage (PCA)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/figures/fig_target_coverage.pdf")
    fig.savefig("results/figures/fig_target_coverage.png", dpi=150)
    print(cov)
    print("Wrote results/target_coverage.csv and figures")


if __name__ == "__main__":
    main()
