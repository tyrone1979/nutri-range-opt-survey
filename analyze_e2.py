#!/usr/bin/env python3
"""Analyze E2 results: summaries, Friedman/Nemenyi-style ranks, figures."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from statistical_tests import friedman_test, nemenyi_test, wilcoxon_pairwise, cohens_d


def load_e2(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["status"] == "ok"].copy() if "status" in df.columns else df
    return df


def aggregate_run_level(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (sparsity, target, algorithm) = mean over reps."""
    keys = ["sparsity", "target_id", "algorithm"]
    ag = df.groupby(keys).agg(
        FHR=("FHR", "mean"),
        BestFit=("BestFit", "mean"),
        FFG_evals=("FFG_evals", "median"),
        FSC=("FSC", "mean"),
        runtime=("runtime", "mean"),
        n_evals=("n_evals", "mean"),
        density=("feasible_density", "first"),
        n_reps=("rep", "nunique"),
    ).reset_index()
    return ag


def summary_tables(df: pd.DataFrame, out_dir: str, tag: str):
    os.makedirs(out_dir, exist_ok=True)
    fhr = df.groupby(["sparsity", "algorithm"])["FHR"].agg(["mean", "std", "median", "count"])
    fhr.to_csv(f"{out_dir}/{tag}_fhr_by_sparsity_algo.csv")
    ffg = df.groupby(["sparsity", "algorithm"])["FFG_evals"].agg(["mean", "median", "std"])
    ffg.to_csv(f"{out_dir}/{tag}_ffg_by_sparsity_algo.csv")
    # wide FHR
    wide = df.groupby(["sparsity", "algorithm"])["FHR"].mean().unstack()
    wide.to_csv(f"{out_dir}/{tag}_fhr_wide.csv")
    return fhr, wide


def build_matrix(ag: pd.DataFrame, sparsity: float, metric: str = "FHR"):
    """targets × algorithms matrix for Friedman."""
    sub = ag[np.isclose(ag["sparsity"], sparsity)]
    piv = sub.pivot(index="target_id", columns="algorithm", values=metric)
    piv = piv.dropna(axis=0, how="any")
    return piv


def stats_for_sparsity(piv: pd.DataFrame) -> dict:
    mat = piv.to_numpy(dtype=float)
    names = list(piv.columns)
    out = {"algorithms": names, "n_targets": int(piv.shape[0])}
    if piv.shape[0] < 3 or piv.shape[1] < 2:
        out["error"] = "insufficient cells"
        return out
    # For FHR higher is better — friedman_test in repo sorts by larger=better via -argsort
    try:
        stat, p = friedman_test(mat)
        out["friedman"] = {"stat": float(stat), "p": float(p)}
    except Exception as e:
        out["friedman"] = {"error": str(e)}
    try:
        out["nemenyi"] = {
            "mean_ranks": nemenyi_test(mat, names)["mean_ranks"],
            "critical_difference": float(nemenyi_test(mat, names)["critical_difference"]),
            "significant_pairs": [
                {"a": a, "b": b, "rank_diff": float(d), "cd": float(cd)}
                for a, b, d, cd in nemenyi_test(mat, names)["significant_pairs"]
            ],
        }
    except Exception as e:
        out["nemenyi"] = {"error": str(e)}
    try:
        pairs = wilcoxon_pairwise(mat, names)
        out["wilcoxon"] = {
            f"{a}__{b}": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                          for k, v in d.items()}
            for (a, b), d in pairs.items()
        }
    except Exception as e:
        out["wilcoxon"] = {"error": str(e)}
    # mean ranks manually
    ranks = np.argsort(np.argsort(-mat, axis=1), axis=1) + 1
    out["mean_ranks"] = dict(zip(names, ranks.mean(axis=0).tolist()))
    return out


def interaction_proxy(ag: pd.DataFrame) -> pd.DataFrame:
    """
    Rank of each algorithm within each (sparsity) averaged over targets.
    Rank variance across sparsity ≈ interaction strength proxy.
    """
    rows = []
    for sp in sorted(ag["sparsity"].unique()):
        piv = build_matrix(ag, sp, "FHR")
        if piv.empty:
            continue
        mat = piv.to_numpy(float)
        ranks = np.argsort(np.argsort(-mat, axis=1), axis=1) + 1
        mean_r = ranks.mean(axis=0)
        for alg, r in zip(piv.columns, mean_r):
            rows.append({"sparsity": sp, "algorithm": alg, "mean_rank": float(r)})
    return pd.DataFrame(rows)


def plot_fhr(wide: pd.DataFrame, out_png: str, title: str):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for col in wide.columns:
        ax.plot(wide.index.astype(float), wide[col], marker="o", label=col)
    ax.set_xlabel("Sparsity factor")
    ax.set_ylabel("Mean FHR")
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    fig.savefig(out_png.replace(".png", ".pdf"))
    plt.close(fig)


def plot_ranks(rank_df: pd.DataFrame, out_png: str, title: str):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    piv = rank_df.pivot(index="sparsity", columns="algorithm", values="mean_rank")
    for col in piv.columns:
        ax.plot(piv.index.astype(float), piv[col], marker="o", label=col)
    ax.set_xlabel("Sparsity factor")
    ax.set_ylabel("Mean rank (1=best)")
    ax.invert_yaxis()
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    fig.savefig(out_png.replace(".png", ".pdf"))
    plt.close(fig)


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--tag", default="e2")
    ap.add_argument("--out-dir", default="results/statistics")
    ap.add_argument("--fig-dir", default="results/figures")
    ap.add_argument("--table-dir", default="results/tables")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)
    os.makedirs(args.table_dir, exist_ok=True)

    df = load_e2(args.input)
    print(f"Loaded {len(df)} rows from {args.input}")
    print(f"class={df['result_class'].unique().tolist() if 'result_class' in df.columns else '?'}")

    fhr, wide = summary_tables(df, args.table_dir, args.tag)
    print("\nFHR mean by sparsity × algorithm:")
    print(wide.round(3))

    ag = aggregate_run_level(df)
    ag.to_csv(f"{args.table_dir}/{args.tag}_agg_target_algo.csv", index=False)

    stats_all = {}
    for sp in sorted(df["sparsity"].unique()):
        piv = build_matrix(ag, float(sp), "FHR")
        stats_all[str(sp)] = stats_for_sparsity(piv)
        print(f"\n=== sparsity={sp} Friedman ===")
        print(json.dumps(_jsonable(stats_all[str(sp)].get("friedman", {})), indent=2))
        print("mean_ranks:", stats_all[str(sp)].get("mean_ranks"))

    with open(f"{args.out_dir}/{args.tag}_stats.json", "w") as f:
        json.dump(_jsonable(stats_all), f, indent=2)

    rank_df = interaction_proxy(ag)
    rank_df.to_csv(f"{args.table_dir}/{args.tag}_mean_ranks.csv", index=False)
    # rank variance across sparsity per algorithm
    if len(rank_df):
        rv = rank_df.groupby("algorithm")["mean_rank"].agg(["mean", "std", "min", "max"])
        rv["range"] = rv["max"] - rv["min"]
        rv = rv.sort_values("range", ascending=False)
        rv.to_csv(f"{args.table_dir}/{args.tag}_rank_stability.csv")
        print("\nRank stability (std/range across sparsity):")
        print(rv.round(3))

    plot_fhr(wide, f"{args.fig_dir}/fig_{args.tag}_fhr_sparsity.png",
             f"{args.tag}: Mean FHR vs sparsity")
    if len(rank_df):
        plot_ranks(rank_df, f"{args.fig_dir}/fig_{args.tag}_rank_sparsity.png",
                   f"{args.tag}: Mean rank vs sparsity (1=best)")

    # budget fairness check
    bud = df.groupby("algorithm")["n_evals"].agg(["min", "median", "max", "mean"])
    bud.to_csv(f"{args.table_dir}/{args.tag}_n_evals_by_algo.csv")
    print("\nn_evals by algorithm:")
    print(bud.round(1))

    meta = {
        "input": args.input,
        "n_rows": len(df),
        "n_targets": int(df["target_id"].nunique()),
        "n_algos": int(df["algorithm"].nunique()),
        "factors": sorted([float(x) for x in df["sparsity"].unique().tolist()]),
        "repeats": int(df["rep"].nunique()) if "rep" in df.columns else None,
        "result_class": str(df["result_class"].iloc[0]) if "result_class" in df.columns else None,
    }
    with open(f"{args.out_dir}/{args.tag}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nWrote stats to {args.out_dir}/ and tables/figures")


if __name__ == "__main__":
    main()
