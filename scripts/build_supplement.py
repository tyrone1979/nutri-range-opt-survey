#!/usr/bin/env python3
"""Build tables + figures for the TEVC supplementary material."""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT_TAB = ROOT / "manuscript" / "supplement_tables"
OUT_FIG = ROOT / "manuscript" / "supplement_figs"
RAW = ROOT / "results" / "raw"
FIGS = ROOT / "results" / "figures"

ALG_ORDER = [
    "MA", "ACMO-Diet", "NSGA2", "BMOEA", "SPEA2",
    "RandomRepair", "RVEA", "CTAEA", "Greedy", "Random",
]
ALG_SHORT = {
    "ACMO-Diet": "ACMO", "NSGA2": "NSGA-II", "CTAEA": "C-TAEA",
    "RandomRepair": "Repair",
}


def short(a: str) -> str:
    return ALG_SHORT.get(a, a)


def ensure_dirs():
    OUT_TAB.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)


def seed_fhr_stats(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Target-average within seed, then mean±std over 10 seeds."""
    g = df.groupby(keys + ["algorithm", "rep"], as_index=False)["FHR"].mean()
    out = g.groupby(keys + ["algorithm"], as_index=False)["FHR"].agg(
        mean="mean", std="std", n="count"
    )
    return out


def fmt_cell(m: float, s: float) -> str:
    if abs(m - 1.0) < 1e-12 and (not np.isfinite(s) or s < 1e-12):
        return r"$\mathbf{1.000}$"
    sm = 0.0 if not np.isfinite(s) else s
    return f"{m:.3f}$\\pm${sm:.3f}"


def write_tex(name: str, body: str):
    (OUT_TAB / name).write_text(body)
    print("wrote", OUT_TAB / name)


# ---------- Part A ----------
def build_s1_s2():
    df = pd.read_csv(ROOT / "data" / "representative_targets.csv")
    # S1: all targets as longtable-friendly tabular pieces
    rows = []
    rows.append(r"\begin{tabular}{@{}clrr@{}}")
    rows.append(r"\toprule")
    rows.append(r"Target & Nutrient & $L_j$ & $U_j$ \\")
    rows.append(r"\midrule")
    for _, r in df.iterrows():
        tid = f"rep\_{int(r['representative_id']):03d}"
        td = json.loads(r["target_json"])
        first = True
        for nut in sorted(td.keys()):
            lo, hi = td[nut]
            hi_s = r"$10^{7}$" if float(hi) >= 1e5 else f"{hi:g}"
            lo_s = f"{lo:g}"
            tgt = tid if first else ""
            nut_tex = nut.replace("_", r"\_")
            rows.append(f"{tgt} & {nut_tex} & {lo_s} & {hi_s} \\\\")
            first = False
        rows.append(r"\midrule")
    rows[-1] = r"\bottomrule"
    rows.append(r"\end{tabular}")
    write_tex("table_s1.tex", "\n".join(rows))

    # S2: scale rep_000 with sparsity_ops using deterministic caps
    from sparsity_ops import prepare_sparsity_target

    td0 = json.loads(df.loc[df.representative_id == 0, "target_json"].iloc[0])
    names = list(td0.keys())
    arr = np.array([[td0[n][0], td0[n][1]] for n in names], dtype=float)
    caps = np.array([
        float(td0[n][1]) if float(td0[n][1]) < 1e5 else 5000.0
        for n in names
    ], dtype=float)

    scaled = {}
    for f in [0.5, 1.0, 1.5]:
        scaled[f], meta = prepare_sparsity_target(arr, f, caps)
        print(f"S2 f={f} meta", meta)

    rows = [r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
            r"Nutrient & $L^{0}$ & $U^{0}$ & $L(0.5)$ & $U(0.5)$ & $L(1.5)$ & $U(1.5)$ \\",
            r"\midrule"]
    for i, n in enumerate(names):
        lo0, hi0 = scaled[1.0][i]
        lo5, hi5 = scaled[0.5][i]
        lo15, hi15 = scaled[1.5][i]

        def fmt(x):
            return f"{x:.1f}" if x < 1e5 else r"$\mathrm{cap}$"

        nut_tex = n.replace("_", r"\_")
        rows.append(
            f"{nut_tex} & {fmt(lo0)} & {fmt(hi0)} & {fmt(lo5)} & {fmt(hi5)} & {fmt(lo15)} & {fmt(hi15)} \\\\"
        )
    rows += [r"\bottomrule", r"\end{tabular}"]
    write_tex("table_s2.tex", "\n".join(rows))


def fig_s1_nutrient_sparsity():
    # Sample NHANES for zero-fraction heatmap-like bar + matrix of top nutrients
    path = ROOT / "data" / "nhanes.csv"
    df = pd.read_csv(path)
    nut_cols = [c for c in df.columns if c not in ("food_id", "grams")]
    # subsample unique foods
    if "food_id" in df.columns:
        df = df.drop_duplicates("food_id")
    if len(df) > 30000:
        df = df.sample(30000, random_state=42)
    X = df[nut_cols].to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=0.0)
    # fraction near-zero (<1e-8 of max or ==0)
    zero_frac = (np.abs(X) < 1e-12).mean(axis=0)
    # log1p mean
    logmean = np.log1p(np.clip(X, 0, None)).mean(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    order = np.argsort(zero_frac)[::-1]
    labels = [nut_cols[i].replace("_", " ") for i in order]
    axes[0].barh(range(len(order)), zero_frac[order], color="#3182bd")
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels(labels, fontsize=7)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Fraction of foods with zero amount")
    axes[0].set_title("(a) Nutrient zero-sparsity")
    axes[0].grid(True, axis="x", ls=":", alpha=0.4)

    # pairwise co-occurrence of nonzero
    B = (np.abs(X) > 1e-12).astype(float)
    # subsample columns if many
    C = (B.T @ B) / B.shape[0]
    im = axes[1].imshow(C, cmap="YlOrRd", vmin=0, vmax=1)
    axes[1].set_xticks(range(len(nut_cols)))
    axes[1].set_yticks(range(len(nut_cols)))
    axes[1].set_xticklabels([c[:6] for c in nut_cols], rotation=90, fontsize=6)
    axes[1].set_yticklabels([c[:8] for c in nut_cols], fontsize=6)
    axes[1].set_title("(b) Nonzero co-occurrence rate")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    fig.tight_layout()
    for d in (OUT_FIG, FIGS):
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / "fig_s1_nutrient_sparsity.pdf", bbox_inches="tight")
        fig.savefig(d / "fig_s1_nutrient_sparsity.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("fig S1 done")


# ---------- Part C FHR tables ----------
def build_fhr_tables():
    e1 = pd.read_csv(RAW / "e1_scaling_full.csv")
    e2 = pd.read_csv(RAW / "e2_sparsity_full.csv")
    e3 = pd.read_csv(RAW / "e3_dimension_full.csv")

    # E1 NHANES + recipe
    nh = e1[e1["dataset"].astype(str).str.contains("nhanes", case=False, na=False)].copy()
    rec = e1[~e1["dataset"].astype(str).str.contains("nhanes", case=False, na=False)].copy()
    s_nh = seed_fhr_stats(nh, ["scale"])
    # recipe: collapse scales
    rec2 = rec.copy()
    rec2["scale"] = -1
    s_rec = seed_fhr_stats(rec2, ["scale"])

    scales = sorted(s_nh["scale"].unique())
    algos = [a for a in ALG_ORDER if a in set(s_nh["algorithm"]) | set(s_rec["algorithm"])]

    def pivot_tex(stats: pd.DataFrame, row_key: str, row_vals, row_labels, caption_note=""):
        lines = []
        header = " & ".join([short(a) for a in algos])
        lines.append(r"\begin{tabular}{@{}l" + "c" * len(algos) + "@{}}")
        lines.append(r"\toprule")
        lines.append(row_key + " & " + header + r" \\")
        lines.append(r"\midrule")
        for rv, lab in zip(row_vals, row_labels):
            cells = []
            for a in algos:
                sub = stats[(stats[row_key if row_key != "Regime" else list(stats.columns)[0]] == rv) & (stats["algorithm"] == a)]
                # fix: pass key name
                pass
        return lines  # placeholder

    # Manual clean builders
    lines = [r"\begin{tabular}{@{}l" + "c" * len(algos) + "@{}}", r"\toprule",
             "Scale & " + " & ".join(short(a) for a in algos) + r" \\", r"\midrule"]
    # recipe row
    cells = []
    for a in algos:
        sub = s_rec[s_rec["algorithm"] == a]
        if sub.empty:
            cells.append("---")
        else:
            cells.append(fmt_cell(sub["mean"].iloc[0], sub["std"].iloc[0]))
    lines.append("Recipe & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    for sc in scales:
        cells = []
        for a in algos:
            sub = s_nh[(s_nh["scale"] == sc) & (s_nh["algorithm"] == a)]
            if sub.empty:
                cells.append("---")
            else:
                cells.append(fmt_cell(sub["mean"].iloc[0], sub["std"].iloc[0]))
        lab = f"{int(sc)//1000}K" if sc < 200000 else r"$\approx$210K"
        if sc == 210620:
            lab = r"$\approx$210K"
        elif sc >= 1000:
            lab = f"{int(sc/1000)}K"
        else:
            lab = str(int(sc))
        lines.append(f"{lab} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_tex("table_s4.tex", "\n".join(lines))

    # E2
    s2 = seed_fhr_stats(e2, ["sparsity"])
    algos2 = [a for a in ALG_ORDER if a in set(s2["algorithm"])]
    lines = [r"\begin{tabular}{@{}l" + "c" * len(algos2) + "c@{}}", r"\toprule",
             r"$f$ & " + " & ".join(short(a) for a in algos2) + r" & Gap($0.5{\to}1.5$) \\", r"\midrule"]
    means = {}
    for f in [0.5, 1.0, 1.5]:
        cells = []
        for a in algos2:
            sub = s2[(s2["sparsity"] == f) & (s2["algorithm"] == a)]
            m, sd = sub["mean"].iloc[0], sub["std"].iloc[0]
            means[(f, a)] = m
            cells.append(fmt_cell(m, sd))
        lines.append(f"{f:.1f} & " + " & ".join(cells) + r" & --- \\")
    # gap row
    gaps = [f"{means[(1.5,a)]-means[(0.5,a)]:.3f}" for a in algos2]
    lines.append(r"\midrule")
    lines.append(r"Gap & " + " & ".join(["---"] * len(algos2)) + " & " + "/".join(
        f"{means[(1.5,a)]-means[(0.5,a)]:.3f}" for a in algos2[:3]
    ) + r" $\ldots$ \\")
    # Better: put gap as final column per row only for f rows - redo
    lines = [r"\begin{tabular}{@{}l" + "c" * len(algos2) + "c@{}}", r"\toprule",
             r"$f$ & " + " & ".join(short(a) for a in algos2) + r" & Mean gap \\", r"\midrule"]
    for f in [0.5, 1.0, 1.5]:
        cells = []
        for a in algos2:
            sub = s2[(s2["sparsity"] == f) & (s2["algorithm"] == a)]
            cells.append(fmt_cell(sub["mean"].iloc[0], sub["std"].iloc[0]))
        if f == 0.5:
            extra = "---"
        elif f == 1.5:
            extra = f"{np.mean([means[(1.5,a)]-means[(0.5,a)] for a in algos2]):.3f}"
        else:
            extra = "---"
        lines.append(f"{f:.1f} & " + " & ".join(cells) + f" & {extra} \\\\")
    # per-algo gap row
    lines.append(r"\midrule")
    gap_cells = [f"{means[(1.5,a)]-means[(0.5,a)]:.3f}" for a in algos2]
    lines.append(r"$1.5{-}0.5$ & " + " & ".join(gap_cells) + r" & --- \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_tex("table_s5.tex", "\n".join(lines))

    # E3 with bold best per row
    s3 = seed_fhr_stats(e3, ["dim"])
    algos3 = [a for a in ALG_ORDER if a in set(s3["algorithm"])]
    lines = [r"\begin{tabular}{@{}l" + "c" * len(algos3) + "@{}}", r"\toprule",
             r"Dim & " + " & ".join(short(a) for a in algos3) + r" \\", r"\midrule"]
    for d in sorted(s3["dim"].unique()):
        vals = {}
        for a in algos3:
            sub = s3[(s3["dim"] == d) & (s3["algorithm"] == a)]
            vals[a] = (sub["mean"].iloc[0], sub["std"].iloc[0])
        best = max(vals[a][0] for a in algos3)
        cells = []
        for a in algos3:
            m, sd = vals[a]
            cell = fmt_cell(m, sd)
            if abs(m - best) < 1e-9:
                cell = r"\textbf{" + cell + "}"
            cells.append(cell)
        lines.append(f"{int(d)} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_tex("table_s6.tex", "\n".join(lines))


# ---------- Part D Wilcoxon ----------
def pairwise_matrix(df: pd.DataFrame, label: str, outfile: str):
    """Wilcoxon signed-rank paired by (target_id, rep)."""
    g = df.copy()
    # Each raw row is already one (target,rep) FHR bit; keep as-is.
    algos = [a for a in ALG_ORDER if a in set(g["algorithm"])]
    piv = g.pivot_table(index=["target_id", "rep"], columns="algorithm", values="FHR", aggfunc="mean")
    sym = np.full((len(algos), len(algos)), r"$\cdot$", dtype=object)
    for i, a in enumerate(algos):
        for j, b in enumerate(algos):
            if i == j:
                sym[i, j] = "---"
                continue
            if i > j:
                continue
            x = piv[a].to_numpy(dtype=float)
            y = piv[b].to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            x, y = x[mask], y[mask]
            diff = x - y
            if len(x) < 5 or np.allclose(diff, 0):
                mark = r"$\approx$"
                sym[i, j] = mark
                sym[j, i] = mark
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    # discard zeros for wilcoxon
                    if np.all(diff == 0):
                        p = 1.0
                    else:
                        p = stats.wilcoxon(x, y, zero_method="wilcox").pvalue
                except ValueError:
                    p = 1.0
            if p > 0.05:
                mark_ab = mark_ba = r"$\approx$"
            else:
                mark_ab = "+" if np.nanmean(x) > np.nanmean(y) else "-"
                mark_ba = "+" if np.nanmean(y) > np.nanmean(x) else "-"
            sym[i, j] = mark_ab
            sym[j, i] = mark_ba
    lines = [r"\begin{tabular}{@{}l" + "c" * len(algos) + "@{}}", r"\toprule",
             " & " + " & ".join(short(a) for a in algos) + r" \\", r"\midrule"]
    for i, a in enumerate(algos):
        lines.append(short(a) + " & " + " & ".join(sym[i]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_tex(outfile, "\n".join(lines))
    P = np.ones((len(algos), len(algos)))
    for i, a in enumerate(algos):
        for j, b in enumerate(algos):
            if i >= j:
                continue
            x = piv[a].to_numpy(dtype=float)
            y = piv[b].to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            x, y = x[mask], y[mask]
            try:
                P[i, j] = P[j, i] = stats.wilcoxon(x, y, zero_method="wilcox").pvalue
            except Exception:
                P[i, j] = P[j, i] = np.nan
    pd.DataFrame(P, index=algos, columns=algos).to_csv(OUT_TAB / outfile.replace(".tex", "_p.csv"))
    print(label, "nonzero diffs example MA vs Random p=", P[algos.index("MA"), algos.index("Random")] if "MA" in algos and "Random" in algos else "n/a")


def build_wilcoxon():
    e2 = pd.read_csv(RAW / "e2_sparsity_full.csv")
    e3 = pd.read_csv(RAW / "e3_dimension_full.csv")
    pairwise_matrix(e2[e2.sparsity == 0.5], "E2 f=0.5", "table_s7.tex")
    pairwise_matrix(e3[e3.dim == 16], "E3 dim16", "table_s8.tex")


# ---------- Part E N* ----------
def fig_s2_unstable_nstar():
    e1 = pd.read_csv(RAW / "e1_scaling_full.csv")
    nh = e1[e1["dataset"].astype(str).str.contains("nhanes", case=False, na=False)]
    s = seed_fhr_stats(nh, ["scale"])
    scales = np.array(sorted(s["scale"].unique()), dtype=float)
    x = np.log10(scales)

    def logistic(xv, L, k, x0):
        return L / (1.0 + np.exp(k * (xv - x0)))

    # four bootstrap splits of targets
    targets = sorted(nh["target_id"].unique())
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2), sharex=True, sharey=True)
    axes = axes.ravel()
    algos_show = ["MA", "NSGA2", "RVEA", "Random"]
    colors = {"MA": "#1b9e77", "NSGA2": "#d95f02", "RVEA": "#7570b3", "Random": "#e7298a"}

    for ax_i, ax in enumerate(axes):
        # resample 8/12 targets
        chosen = rng.choice(targets, size=min(8, len(targets)), replace=False)
        sub = nh[nh["target_id"].isin(chosen)]
        ss = seed_fhr_stats(sub, ["scale"])
        for algo in algos_show:
            ys, yerr = [], []
            for sc in scales:
                row = ss[(ss["scale"] == sc) & (ss["algorithm"] == algo)]
                if row.empty:
                    ys.append(np.nan); yerr.append(np.nan)
                else:
                    ys.append(row["mean"].iloc[0]); yerr.append(row["std"].iloc[0])
            ys = np.asarray(ys, float)
            ax.errorbar(x, ys, yerr=yerr, fmt="o-", ms=3, lw=1, color=colors[algo],
                        label=short(algo), alpha=0.9)
            # try fit
            mask = np.isfinite(ys)
            if mask.sum() >= 4:
                try:
                    from scipy.optimize import curve_fit
                    popt, _ = curve_fit(logistic, x[mask], ys[mask], p0=[1, 1, 4], maxfev=5000)
                    xx = np.linspace(x.min(), x.max(), 100)
                    ax.plot(xx, logistic(xx, *popt), ls="--", color=colors[algo], alpha=0.5, lw=1)
                    nstar = 10 ** popt[2]
                    ax.axvline(popt[2], color=colors[algo], ls=":", alpha=0.3)
                    ax.text(0.02, 0.08 + 0.12 * algos_show.index(algo), f"{short(algo)} $N^*\\!\\approx\\!{nstar:.1e}$",
                            transform=ax.transAxes, fontsize=6, color=colors[algo])
                except Exception:
                    ax.text(0.02, 0.08 + 0.12 * algos_show.index(algo), f"{short(algo)} fit fail",
                            transform=ax.transAxes, fontsize=6, color=colors[algo])
        ax.set_title(f"Split {ax_i+1}: targets {len(chosen)}/{len(targets)}")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, ls=":", alpha=0.4)
        if ax_i % 2 == 0:
            ax.set_ylabel("FHR")
        if ax_i >= 2:
            ax.set_xlabel(r"$\log_{10} N$")
    axes[0].legend(fontsize=7, frameon=False, loc="lower left")
    fig.suptitle("Exploratory logistic fits under four target splits (unstable $N^*$)", fontsize=10)
    fig.tight_layout()
    for d in (OUT_FIG, FIGS):
        fig.savefig(d / "fig_s2_unstable_nstar.pdf", bbox_inches="tight")
        fig.savefig(d / "fig_s2_unstable_nstar.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("fig S2 done")


# ---------- Part F FFG ----------
def fig_s3_ffg():
    e2 = pd.read_csv(RAW / "e2_sparsity_full.csv")
    hard = e2[e2.sparsity == 0.5].copy()
    algos = ["MA", "ACMO-Diet", "NSGA2", "BMOEA", "RVEA"]
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    data, labels, notes = [], [], []
    for a in algos:
        g = hard[hard.algorithm == a]
        # success: FHR==1
        hit = g[g["FHR"] >= 0.5]["FFG_evals"].to_numpy()
        n_succ = int((g["FHR"] >= 0.5).sum())
        n_tot = len(g)
        data.append(hit if len(hit) else np.array([np.nan]))
        labels.append(short(a))
        med = np.median(hit) if len(hit) else np.nan
        notes.append(f"{short(a)}: {n_succ}/{n_tot}, med={med:.0f}" if len(hit) else f"{short(a)}: 0/{n_tot}")
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor("#c6dbef")
    # swarm-like overlay
    rng = np.random.default_rng(0)
    for i, d in enumerate(data, start=1):
        d = d[np.isfinite(d)]
        if len(d) == 0:
            continue
        jitter = rng.normal(0, 0.06, size=len(d))
        ax.scatter(np.full(len(d), i) + jitter, d, s=8, alpha=0.35, color="#08519c")
    ax.set_ylabel("FFG evaluations (hits only)")
    ax.set_xlabel("Algorithm")
    ax.set_title("E2 $f=0.5$: evaluations to first feasible")
    ax.set_yscale("log")
    ax.grid(True, axis="y", ls=":", alpha=0.4)
    note = "; ".join(notes)
    fig.text(0.5, -0.02, note, ha="center", fontsize=6, wrap=True)
    fig.tight_layout()
    for d in (OUT_FIG, FIGS):
        fig.savefig(d / "fig_s3_ffg_first_hit.pdf", bbox_inches="tight")
        fig.savefig(d / "fig_s3_ffg_first_hit.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("fig S3 done", notes)


# ---------- Part G budgets ----------
def build_s9():
    rows_data = []
    for camp, path in [("E1", "e1_scaling_full.csv"), ("E2", "e2_sparsity_full.csv"), ("E3", "e3_dimension_full.csv")]:
        df = pd.read_csv(RAW / path)
        med = df.groupby("algorithm")["n_evals"].median()
        for a, v in med.items():
            rows_data.append({"campaign": camp, "algorithm": a, "median_n_evals": v})
    wide = pd.DataFrame(rows_data).pivot(index="algorithm", columns="campaign", values="median_n_evals")
    for c in ["E1", "E2", "E3"]:
        if c not in wide.columns:
            wide[c] = np.nan
    wide = wide[["E1", "E2", "E3"]]
    algos = [a for a in ALG_ORDER if a in wide.index]
    lines = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
             r"Algorithm & E1 median $n_{\mathrm{evals}}$ & E2 & E3 \\", r"\midrule",
             r"Theoretical ceiling & 100000 & 100000 & 100000 \\", r"\midrule"]
    for a in algos:
        e1 = wide.loc[a, "E1"]; e2 = wide.loc[a, "E2"]; e3 = wide.loc[a, "E3"]
        def f(x):
            return "---" if not np.isfinite(x) else f"{x:.0f}"
        lines.append(f"{short(a)} & {f(e1)} & {f(e2)} & {f(e3)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_tex("table_s9.tex", "\n".join(lines))
    # residual percent
    for a in algos:
        for c in ["E1", "E2", "E3"]:
            v = wide.loc[a, c]
            if np.isfinite(v):
                pct = 100 * (v - 100000) / 100000
                if abs(pct) > 0.05:
                    print(f"residual {a} {c}: {pct:.3f}%")


def main():
    ensure_dirs()
    build_s1_s2()
    fig_s1_nutrient_sparsity()
    build_fhr_tables()
    build_wilcoxon()
    fig_s2_unstable_nstar()
    fig_s3_ffg()
    build_s9()
    print("all supplement artifacts ready")


if __name__ == "__main__":
    main()
