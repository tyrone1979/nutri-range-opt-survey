#!/usr/bin/env python3
"""Recompute primary statistics and figures for the submission suite.

ACMO-Diet and BMOEA are outside the empirical comparison. C-TAEA is excluded
because the archived runs used 13 reference directions while their evaluation
counter advanced by the requested population of 500. RVEA is excluded because
the archived implementation terminated after far fewer objective evaluations
than the common ceiling. Those runs therefore did not receive comparable
evaluation budgets.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"
TAB = ROOT / "results" / "tables"
FIG = ROOT / "results" / "figures"
STAT = ROOT / "results" / "statistics"
FIGS = ROOT / "figs"

EXCLUDE = {"ACMO-Diet", "BMOEA", "CTAEA", "RVEA"}
# Display / plot order for the final suite.
ALG_ORDER = [
    "MA",
    "NSGA2",
    "SPEA2",
    "Greedy",
    "RandomRepair",
    "Random",
]
SHORT = {
    "MA": "MA",
    "NSGA2": "NSGA-II",
    "SPEA2": "SPEA2",
    "Greedy": "Greedy",
    "RandomRepair": "Repair",
    "Random": "Random",
}
COLORS = {
    "MA": "#1b9e77",
    "NSGA2": "#d95f02",
    "SPEA2": "#7570b3",
    "Greedy": "#e6ab02",
    "RandomRepair": "#a6761d",
    "Random": "#666666",
}


def filter_algs(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["algorithm"].isin(ALG_ORDER)].copy()


def seed_mean_fhr(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Seed-level mean FHR over targets, then mean/std across the 10 reps."""
    # PRIMARY CSVs use `rep` (0..9) as the matched seed index; `seed` hashes are unique per row.
    g1 = (
        df.groupby(group_cols + ["algorithm", "rep"], as_index=False)["FHR"]
        .mean()
        .rename(columns={"FHR": "seed_FHR"})
    )
    g2 = (
        g1.groupby(group_cols + ["algorithm"], as_index=False)
        .agg(mean=("seed_FHR", "mean"), std=("seed_FHR", "std"), n=("seed_FHR", "count"))
    )
    return g2


def friedman_block_ranks(df: pd.DataFrame, block_cols: list[str] | None = None, higher_better: bool = True):
    """Mean Friedman ranks with targets as matched blocks (seed-averaged FHR)."""
    # Manuscript protocol: average seeds within each target, then Friedman on targets × algorithms.
    ag = df.groupby(["target_id", "algorithm"], as_index=False)["FHR"].mean()
    piv = ag.pivot(index="target_id", columns="algorithm", values="FHR")
    piv = piv.dropna(axis=0, how="any")
    if piv.shape[0] < 2 or piv.shape[1] < 2:
        return {}, np.nan, np.nan, piv
    mat = piv.to_numpy(dtype=float)
    ranks = np.zeros_like(mat)
    for i in range(mat.shape[0]):
        ranks[i] = stats.rankdata(-mat[i], method="average")
    mean_ranks = {c: float(ranks[:, j].mean()) for j, c in enumerate(piv.columns)}
    try:
        stat, p = stats.friedmanchisquare(*[mat[:, j] for j in range(mat.shape[1])])
        # Note: friedmanchisquare tests equality of distributions; ranking uses higher-better above.
        return mean_ranks, float(stat), float(p), piv
    except Exception:
        return mean_ranks, np.nan, np.nan, piv


def add_holm_adjusted_p(records: dict, raw_key: str = "friedman_p") -> None:
    """Add Holm-adjusted p-values to one family of cell-level tests."""
    valid = [
        (float(info[raw_key]), cell)
        for cell, info in records.items()
        if np.isfinite(info.get(raw_key, np.nan))
    ]
    running = 0.0
    m = len(valid)
    for rank, (p, cell) in enumerate(sorted(valid), start=1):
        running = max(running, (m - rank + 1) * p)
        records[cell]["friedman_p_holm"] = min(1.0, running)


def plot_lines(x, series: dict, xlabel, ylabel, title, out_stem, xticklabels=None):
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Nimbus Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
        }
    )
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    for alg in ALG_ORDER:
        if alg not in series:
            continue
        y = series[alg]
        ax.plot(x, y, marker="o", ms=3.5, lw=1.1, color=COLORS.get(alg, None), label=SHORT[alg])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if xticklabels is not None:
        ax.set_xticks(x)
        ax.set_xticklabels(xticklabels)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(frameon=False, ncol=2, loc="best", columnspacing=0.8, handlelength=1.4)
    fig.tight_layout(pad=0.3)
    for d in (FIG, FIGS):
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / f"{out_stem}.pdf", bbox_inches="tight", pad_inches=0.02)
        fig.savefig(d / f"{out_stem}.png", dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def recompute_e1():
    df = filter_algs(pd.read_csv(RAW / "e1_scaling_full.csv"))
    corrected_path = RAW / "swevo_e1_pymoo_corrected.csv"
    if corrected_path.exists():
        corrected = filter_algs(pd.read_csv(corrected_path))
        replace_algs = set(corrected["algorithm"].unique())
        is_nhanes = df["dataset"].astype(str).str.contains("nhanes", case=False, na=False)
        df = pd.concat(
            [df[~(is_nhanes & df["algorithm"].isin(replace_algs))], corrected],
            ignore_index=True,
        )
    nh = df[df["dataset"].astype(str).str.contains("nhanes", case=False, na=False)].copy()

    # Seed-level means by scale
    s_nh = seed_mean_fhr(nh, ["scale"])
    s_nh.to_csv(TAB / "e1_8m_fhr_by_scale_algo.csv", index=False)

    scales = sorted(s_nh["scale"].unique())
    # Overall mean FHR across algorithms per scale (equal weight algs)
    overall = []
    for sc in scales:
        sub = s_nh[s_nh["scale"] == sc]
        overall.append({"scale": sc, "mean_FHR": float(sub["mean"].mean())})
    pd.DataFrame(overall).to_csv(TAB / "e1_8m_overall_by_scale.csv", index=False)

    # Friedman per NHANES scale (blocks = target_id over seeds averaged first)
    fried = {}
    for sc in scales:
        sub = nh[nh["scale"] == sc]
        ranks, stat, p, _ = friedman_block_ranks(sub)
        fried[str(int(sc))] = {"mean_ranks": ranks, "friedman_p": p, "stat": stat}
    add_holm_adjusted_p(fried)

    # Plot
    series = {}
    for alg in ALG_ORDER:
        ys = []
        for sc in scales:
            row = s_nh[(s_nh["scale"] == sc) & (s_nh["algorithm"] == alg)]
            ys.append(float(row["mean"].iloc[0]) if len(row) else np.nan)
        series[alg] = ys
    labs = []
    for sc in scales:
        if sc >= 200000:
            labs.append(r"$\approx$210K")
        else:
            labs.append(f"{int(sc/1000)}K")
    plot_lines(
        list(range(len(scales))),
        series,
        "NHANES food-space scale",
        "Mean FHR",
        "E1: mean FHR versus NHANES scale (6 methods)",
        "fig_e1_full_fhr_scale",
        xticklabels=labs,
    )

    report = {
        "nhanes_overall_by_scale": overall,
        "friedman": fried,
        "band": [min(o["mean_FHR"] for o in overall), max(o["mean_FHR"] for o in overall)],
    }
    (STAT / "E1_8METHOD_SUMMARY.json").write_text(json.dumps(report, indent=2, default=float))
    print("E1 band", report["band"])
    return report


def recompute_e2():
    df = filter_algs(pd.read_csv(RAW / "e2_sparsity_full.csv"))
    s = seed_mean_fhr(df, ["sparsity"])
    s.to_csv(TAB / "e2_8m_fhr_by_sparsity_algo.csv", index=False)

    factors = [0.5, 1.0, 1.5]
    overall = []
    fried = {}
    for f in factors:
        sub = s[np.isclose(s["sparsity"], f)]
        overall.append({"f": f, "mean_FHR": float(sub["mean"].mean())})
        raw = df[np.isclose(df["sparsity"], f)]
        ranks, stat, p, _ = friedman_block_ranks(raw)
        fried[str(f)] = {"mean_ranks": ranks, "friedman_p": p, "stat": stat}
    add_holm_adjusted_p(fried)

    # hard cell f=0.5 table
    hard = s[np.isclose(s["sparsity"], 0.5)].copy()
    # ranks at f=0.5 from friedman mean ranks
    hard_ranks = fried["0.5"]["mean_ranks"]
    hard["rank"] = hard["algorithm"].map(hard_ranks)
    hard = hard.sort_values("rank")
    hard.to_csv(TAB / "e2_8m_f05_ranks.csv", index=False)

    series = {}
    for alg in ALG_ORDER:
        ys = []
        for f in factors:
            row = s[np.isclose(s["sparsity"], f) & (s["algorithm"] == alg)]
            ys.append(float(row["mean"].iloc[0]) if len(row) else np.nan)
        series[alg] = ys
    plot_lines(
        factors,
        series,
        "Interval-width factor $f$",
        "Mean FHR",
        "E2: mean FHR versus interval tightness (6 methods)",
        "fig_e2_full_fhr_sparsity",
        xticklabels=["0.5", "1.0", "1.5"],
    )

    report = {"overall": overall, "friedman": fried, "hard_f05": hard.to_dict(orient="records")}
    (STAT / "E2_8METHOD_SUMMARY.json").write_text(json.dumps(report, indent=2, default=float))
    print("E2 overall", overall)
    print("E2 f=0.5 ranks", sorted(hard_ranks.items(), key=lambda kv: kv[1]))
    return report


def recompute_e3():
    df = filter_algs(pd.read_csv(RAW / "e3_dimension_full.csv"))
    corrected_path = RAW / "swevo_e3_pymoo_corrected.csv"
    if corrected_path.exists():
        corrected = filter_algs(pd.read_csv(corrected_path))
        replace_algs = set(corrected["algorithm"].unique())
        df = pd.concat(
            [df[~df["algorithm"].isin(replace_algs)], corrected],
            ignore_index=True,
        )
    s = seed_mean_fhr(df, ["dim"])
    s.to_csv(TAB / "e3_8m_fhr_by_dim_algo.csv", index=False)
    dims = sorted(s["dim"].unique())
    overall = []
    fried = {}
    for d in dims:
        sub = s[s["dim"] == d]
        overall.append({"dim": int(d), "mean_FHR": float(sub["mean"].mean())})
        raw = df[df["dim"] == d]
        ranks, stat, p, _ = friedman_block_ranks(raw)
        fried[str(int(d))] = {"mean_ranks": ranks, "friedman_p": p}
    add_holm_adjusted_p(fried)

    series = {}
    for alg in [a for a in ALG_ORDER if a in set(s["algorithm"])]:
        ys = [float(s[(s["dim"] == d) & (s["algorithm"] == alg)]["mean"].iloc[0]) for d in dims]
        series[alg] = ys
    plot_lines(
        dims,
        series,
        "Constraint dimensionality $|J|$",
        "Mean FHR",
        "E3: mean FHR versus constraint dimension (4 methods)",
        "fig_e3_full_fhr_dim",
        xticklabels=[str(int(d)) for d in dims],
    )
    # dim-16 ordering
    d16 = s[s["dim"] == 16].sort_values("mean", ascending=False)
    report = {"overall": overall, "friedman": fried, "dim16": d16.to_dict(orient="records")}
    (STAT / "E3_8METHOD_SUMMARY.json").write_text(json.dumps(report, indent=2, default=float))
    print("E3 overall", overall)
    print("E3 dim16", list(zip(d16["algorithm"], d16["mean"].round(3))))
    return report


def recompute_e4(e1r, e2r, e3r):
    """Cross-regime mean Friedman ranks + Kendall tau on shared algorithms."""
    # Per-regime mean ranks: average Friedman ranks over cells in each campaign.
    def avg_ranks(fried_dict):
        # fried_dict values have mean_ranks per cell
        acc = {}
        n = {}
        for cell, info in fried_dict.items():
            for a, r in info["mean_ranks"].items():
                if a in EXCLUDE:
                    continue
                acc[a] = acc.get(a, 0.0) + r
                n[a] = n.get(a, 0) + 1
        return {a: acc[a] / n[a] for a in acc}

    r1 = avg_ranks(e1r["friedman"])
    r2 = avg_ranks(e2r["friedman"])
    r3 = avg_ranks(e3r["friedman"])

    shared = sorted(set(r1) & set(r2) & set(r3), key=lambda a: ALG_ORDER.index(a) if a in ALG_ORDER else 99)
    print("E4 shared", shared)

    rows = []
    for a in ALG_ORDER:
        if a not in r1 and a not in r2 and a not in r3:
            continue
        e1 = r1.get(a, np.nan)
        e2 = r2.get(a, np.nan)
        e3 = r3.get(a, np.nan)
        vals = [v for v in (e1, e2, e3) if np.isfinite(v)]
        overall = float(np.mean(vals)) if vals else np.nan
        rows.append(
            {
                "algorithm": a,
                "E1_NHANES": e1,
                "E2_sparsity": e2,
                "E3_dimension": e3,
                "overall": overall,
            }
        )
    ranks_df = pd.DataFrame(rows).sort_values("overall")
    ranks_df.to_csv(TAB / "e4_cross_regime_mean_ranks.csv", index=False)

    # Hard-cell f=0.5 FHR for table
    hard = pd.read_csv(TAB / "e2_8m_f05_ranks.csv")
    hard_map = {r.algorithm: (float(r.mean), float(r.std), float(r.rank)) for r in hard.itertuples()}

    def kendall(ra, rb):
        xa = np.array([ra[a] for a in shared], dtype=float)
        xb = np.array([rb[a] for a in shared], dtype=float)
        tau, p = stats.kendalltau(xa, xb)
        return float(tau), float(p)

    t12, p12 = kendall(r1, r2)
    t13, p13 = kendall(r1, r3)
    t23, p23 = kendall(r2, r3)
    tau_raw = {"E1_E2": p12, "E1_E3": p13, "E2_E3": p23}
    tau_holm = {}
    running = 0.0
    for rank, (name, p) in enumerate(
        sorted(tau_raw.items(), key=lambda item: item[1]), start=1
    ):
        running = max(running, (len(tau_raw) - rank + 1) * p)
        tau_holm[name] = min(1.0, running)
    report = {
        "shared": shared,
        "tau": {"E1_E2": (t12, p12), "E1_E3": (t13, p13), "E2_E3": (t23, p23)},
        "tau_p_holm": tau_holm,
        "ranks": ranks_df.to_dict(orient="records"),
        "hard_f05": hard_map,
    }
    (STAT / "E4_8METHOD_SUMMARY.json").write_text(json.dumps(report, indent=2, default=float))
    print("E4 Kendall", report["tau"])
    print(ranks_df.to_string(index=False))

    # Heatmap of ranks (shared only, regimes as rows)
    plt.rcParams.update({"font.family": "serif", "font.size": 8})
    mat = np.array([[r1[a], r2[a], r3[a]] for a in shared], dtype=float).T  # 3 x n
    fig, ax = plt.subplots(figsize=(4.2, 2.2))
    im = ax.imshow(mat, aspect="auto", cmap="viridis_r")
    ax.set_xticks(range(len(shared)))
    ax.set_xticklabels([SHORT.get(a, a) for a in shared], rotation=35, ha="right")
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["E1", "E2", "E3"])
    ax.set_title("Cross-regime mean Friedman ranks")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color="white", fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Mean rank (lower better)")
    fig.tight_layout(pad=0.3)
    for d in (FIG, FIGS):
        fig.savefig(d / "rank_concordance_heatmap.pdf", bbox_inches="tight")
        fig.savefig(d / "rank_concordance_heatmap.png", dpi=220, bbox_inches="tight")
        fig.savefig(d / "fig_e4_cross_regime_ranks.pdf", bbox_inches="tight")
        fig.savefig(d / "fig_e4_cross_regime_ranks.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return report


def recompute_budget():
    raw = RAW / "e_budget_f05_corrected.csv"
    if not raw.exists():
        print("budget raw missing, skip")
        return None
    df = pd.read_csv(raw)
    keep = ["MA", "NSGA2", "RandomRepair", "Random"]
    df = df[df["algorithm"].isin(keep)].copy()
    g = df.groupby(["max_evaluations", "algorithm"], as_index=False).agg(
        mean_FHR=("FHR", "mean"), std_FHR=("FHR", "std"), n=("FHR", "count")
    )
    g["ci95"] = 1.96 * g["std_FHR"] / g["n"].clip(lower=1) ** 0.5
    g.to_csv(TAB / "e_budget_fhr_by_budget_algo.csv", index=False)

    plt.rcParams.update({"font.family": "serif", "font.size": 7, "axes.titlesize": 8, "legend.fontsize": 6})
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    for alg in keep:
        ag = g[g["algorithm"] == alg].sort_values("max_evaluations")
        ax.errorbar(
            ag["max_evaluations"], ag["mean_FHR"], yerr=ag["ci95"],
            marker="o", label=SHORT.get(alg, alg), lw=1.0, ms=3, capsize=1.5,
            color=COLORS.get(alg),
        )
    ax.set_xlabel("Evaluation budget")
    ax.set_ylabel("Mean FHR")
    ax.set_title(r"Budget sensitivity (NHANES-10K, $f=0.5$)")
    ax.set_xticks([25000, 50000, 100000, 200000])
    ax.set_xticklabels(["25k", "50k", "100k", "200k"])
    ax.set_ylim(0.0, 0.36)
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(loc="upper left", ncol=2, frameon=False, columnspacing=0.7, handlelength=1.3)
    fig.tight_layout(pad=0.25)
    for d in (FIG, FIGS):
        fig.savefig(d / "fig_budget_vs_fhr.pdf", bbox_inches="tight", pad_inches=0.02)
        fig.savefig(d / "fig_budget_vs_fhr.png", dpi=220, bbox_inches="tight", pad_inches=0.02)
        # also budget_vs_fhr alias used by some paths
        fig.savefig(d / "budget_vs_fhr.pdf", bbox_inches="tight", pad_inches=0.02)
        fig.savefig(d / "budget_vs_fhr.png", dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    # Means for manuscript text
    means = {}
    for b in [25000, 50000, 100000, 200000]:
        sub = g[g["max_evaluations"] == b]
        means[b] = {
            "per_algo": {r.algorithm: float(r.mean_FHR) for r in sub.itertuples()},
            "six_mean_legacy_note": "now 4-method mean",
            "mean4": float(sub["mean_FHR"].mean()),
        }
    (STAT / "BUDGET_8METHOD_SUMMARY.json").write_text(json.dumps(means, indent=2))
    print("budget means", {k: v["mean4"] for k, v in means.items()})
    return means


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    STAT.mkdir(parents=True, exist_ok=True)
    e1 = recompute_e1()
    e2 = recompute_e2()
    e3 = recompute_e3()
    e4 = recompute_e4(e1, e2, e3)
    bud = recompute_budget()
    summary = {"e1": e1, "e2": e2["overall"], "e3": e3["overall"], "e4": e4["tau"], "budget": bud}
    (STAT / "SUITE8_RECOMPUTE_SUMMARY.json").write_text(json.dumps(summary, indent=2, default=float))
    print("wrote SUITE8_RECOMPUTE_SUMMARY.json")


if __name__ == "__main__":
    main()
