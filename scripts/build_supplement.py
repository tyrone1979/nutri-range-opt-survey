#!/usr/bin/env python3
"""Build tables and figures for the SWEO supplementary material."""
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
OUT_TAB = ROOT / "manuscript" / "swevo" / "supplement_tables"
OUT_FIG = ROOT / "manuscript" / "swevo" / "supplement_figs"
RAW = ROOT / "results" / "raw"
FIGS = ROOT / "results" / "figures"

ALG_ORDER = [
    "MA", "NSGA2", "SPEA2",
    "Greedy", "RandomRepair", "Random",
]
ALG_SHORT = {
    "NSGA2": "NSGA-II",
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


def load_e1_submission() -> pd.DataFrame:
    df = pd.read_csv(RAW / "e1_scaling_full.csv")
    corrected_path = RAW / "swevo_e1_pymoo_corrected.csv"
    if corrected_path.exists():
        corrected = pd.read_csv(corrected_path)
        replace_algs = set(corrected["algorithm"].unique())
        is_nhanes = df["dataset"].astype(str).str.contains(
            "nhanes", case=False, na=False
        )
        df = pd.concat(
            [df[~(is_nhanes & df["algorithm"].isin(replace_algs))], corrected],
            ignore_index=True,
        )
    return df[df["algorithm"].isin(ALG_ORDER)].copy()


def load_e3_submission() -> pd.DataFrame:
    df = pd.read_csv(RAW / "e3_dimension_full.csv")
    corrected_path = RAW / "swevo_e3_pymoo_corrected.csv"
    if corrected_path.exists():
        corrected = pd.read_csv(corrected_path)
        replace_algs = set(corrected["algorithm"].unique())
        df = pd.concat(
            [df[~df["algorithm"].isin(replace_algs)], corrected],
            ignore_index=True,
        )
    return df[df["algorithm"].isin(ALG_ORDER)].copy()


def fmt_cell(m: float, s: float) -> str:
    if abs(m - 1.0) < 1e-12 and (not np.isfinite(s) or s < 1e-12):
        return r"$\mathbf{1.000}$"
    sm = 0.0 if not np.isfinite(s) else s
    return f"{m:.3f}$\\pm${sm:.3f}"


def tab_begin(colspec: str) -> str:
    """Content-aware tabular: pass a full column spec (no outer @{})."""
    return rf"\begin{{tabular}}{{@{{}}{colspec}@{{}}}}"


def tabx_begin(colspec: str, width: str = r"\textwidth") -> str:
    """tabularx with explicit (non-equal) column spec; use X only on text-heavy cols."""
    return rf"\begin{{tabularx}}{{{width}}}{{@{{}}{colspec}@{{}}}}"


def tab_end() -> str:
    return r"\end{tabular}"


def tabx_end() -> str:
    return r"\end{tabularx}"


# Shared matplotlib typography for all supplement figures (serif to match IEEEtran body).
SUP_RC = {
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.titlesize": 10,
    "axes.linewidth": 0.7,
}


def write_tex(name: str, body: str):
    (OUT_TAB / name).write_text(body)
    print("wrote", OUT_TAB / name)


# ---------- Part A ----------
def build_s1_s2():
    df = pd.read_csv(ROOT / "data" / "representative_targets.csv")
    # S1: breakable longtable with content-aware widths (Nutrient widest).
    caption = (
        r"Nutrient interval boxes for representative targets "
        r"\texttt{rep\_000}--\texttt{rep\_011}. "
        r"An upper bound of 99{,}999 or larger marks a one-sided face."
    )
    rows = [
        r"\begin{longtable}{@{}>{\raggedright\arraybackslash}p{0.16\textwidth}"
        r">{\raggedright\arraybackslash}p{0.44\textwidth}"
        r">{\raggedleft\arraybackslash}p{0.18\textwidth}"
        r">{\raggedleft\arraybackslash}p{0.18\textwidth}@{}}",
        rf"\caption{{{caption}}}",
        r"\label{tab:s1}\\",
        r"\toprule",
        r"Target & Nutrient & $L_j$ & $U_j$ \\",
        r"\midrule",
        r"\endfirsthead",
        r"\caption[]{Nutrient interval boxes (continued).}\\",
        r"\toprule",
        r"Target & Nutrient & $L_j$ & $U_j$ \\",
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{4}{r}{\emph{Continued on next page}}\\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for _, r in df.iterrows():
        tid = f"rep\\_{int(r['representative_id']):03d}"
        td = json.loads(r["target_json"])
        first = True
        for nut in sorted(td.keys()):
            lo, hi = td[nut]
            hi_s = r"$\mathrm{sentinel}$" if float(hi) >= 99_999 else f"{hi:g}"
            lo_s = f"{lo:g}"
            tgt = tid if first else ""
            nut_tex = nut.replace("_", r"\_")
            rows.append(f"{tgt} & {nut_tex} & {lo_s} & {hi_s} \\\\")
            first = False
        rows.append(r"\midrule")
    if rows[-1].strip() == r"\midrule":
        rows.pop()
    rows.append(r"\end{longtable}")
    write_tex("table_s1.tex", "\n".join(rows))

    # S2: reproduce the E2 sanitization and scaling for rep_000.
    import common as cm
    from e1_scaling import load_nhanes_full, sample_food_space
    from run_benchmark import align_target_to_space, load_representative_targets
    from sparsity_ops import empirical_meal_caps, prepare_sparsity_target

    matrix, meal_tags, nutrients, _ = cm.load_data(
        "data/nhanes.csv", "data/food_type.csv", "data/target.csv"
    )
    full = load_nhanes_full()
    full.nutrients = matrix.astype(np.float32)
    full.meal_tags = meal_tags.astype(np.int8)
    full.food_ids = np.arange(len(matrix), dtype=np.int32)
    fs = sample_food_space(full, 10_000, seed=42)
    target = load_representative_targets("data/representative_targets.csv", 1)[0]
    aligned = align_target_to_space(target, nutrients)
    if aligned is None:
        raise RuntimeError("rep_000 does not align with the NHANES nutrient matrix")
    names = list(aligned.nutrient_names)
    arr = np.asarray(aligned.target_array, dtype=float)
    caps = empirical_meal_caps(
        fs.nutrients,
        aligned.col_idx,
        fs.food_ids,
        fs.meal_tags,
        fs.min_foods,
        fs.max_foods,
        True,
        n_samples=600,
        percentile=99.0,
        seed=42 + aligned.n_dims,
    )

    scaled = {}
    for f in [0.5, 1.0, 1.5]:
        scaled[f], meta = prepare_sparsity_target(arr, f, caps)
        print(f"S2 f={f} meta", meta)

    rows = [tabx_begin(r">{\raggedright\arraybackslash}p{0.20\textwidth}*{6}{>{\centering\arraybackslash}X}"), r"\toprule",
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
    rows += [r"\bottomrule", tabx_end()]
    write_tex("table_s2.tex", "\n".join(rows))


def fig_s1_nutrient_sparsity():
    # Sample NHANES for zero-fraction heatmap-like bar + matrix of top nutrients
    plt.rcParams.update(SUP_RC)
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
    axes[0].set_yticklabels(labels)
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
    axes[1].set_xticklabels([c[:6] for c in nut_cols], rotation=90)
    axes[1].set_yticklabels([c[:8] for c in nut_cols])
    axes[1].set_title("(b) Nonzero co-occurrence rate")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    fig.tight_layout(pad=0.4)
    for d in (OUT_FIG, FIGS):
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / "fig_s1_nutrient_sparsity.pdf", bbox_inches="tight")
        fig.savefig(d / "fig_s1_nutrient_sparsity.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("fig S1 done")


# ---------- Part C FHR tables ----------
def build_fhr_tables():
    e1 = load_e1_submission()
    e2 = pd.read_csv(RAW / "e2_sparsity_full.csv")
    e3 = load_e3_submission()

    # E1 nested survey-derived pools
    nh = e1[e1["dataset"].astype(str).str.contains("nhanes", case=False, na=False)].copy()
    s_nh = seed_fhr_stats(nh, ["scale"])

    scales = sorted(s_nh["scale"].unique())
    algos = [a for a in ALG_ORDER if a in set(s_nh["algorithm"])]

    # Manual clean builders
    lines = [tab_begin("l " + " ".join(["c"] * len(algos))), r"\toprule",
             "Scale & " + " & ".join(short(a) for a in algos) + r" \\", r"\midrule"]
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
    lines += [r"\bottomrule", tab_end()]
    write_tex("table_s4.tex", "\n".join(lines))

    # E2
    s2 = seed_fhr_stats(e2, ["sparsity"])
    algos2 = [a for a in ALG_ORDER if a in set(s2["algorithm"])]
    lines = [tab_begin("c " + " ".join(["c"] * (len(algos2) + 1))), r"\toprule",
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
    lines = [tab_begin("c " + " ".join(["c"] * (len(algos2) + 1))), r"\toprule",
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
    lines += [r"\bottomrule", tab_end()]
    write_tex("table_s5.tex", "\n".join(lines))

    # E3 with bold best per row
    s3 = seed_fhr_stats(e3, ["dim"])
    algos3 = [a for a in ALG_ORDER if a in set(s3["algorithm"])]
    lines = [tab_begin("c " + " ".join(["c"] * len(algos3))), r"\toprule",
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
    lines += [r"\bottomrule", tab_end()]
    write_tex("table_s6.tex", "\n".join(lines))


# ---------- Part D Wilcoxon ----------
def pairwise_matrix(df: pd.DataFrame, label: str, outfile: str):
    """Target-blocked Wilcoxon tests with Holm family-wise correction."""
    # Match the primary Friedman analysis: average the ten seeds within each
    # target, then treat targets (rather than individual runs) as blocks.
    g = (
        df.groupby(["target_id", "algorithm"], as_index=False)["FHR"]
        .mean()
    )
    algos = [a for a in ALG_ORDER if a in set(g["algorithm"])]
    piv = g.pivot(index="target_id", columns="algorithm", values="FHR")
    n_alg = len(algos)
    raw = np.ones((n_alg, n_alg))
    pairs = []
    for i, a in enumerate(algos):
        for j in range(i + 1, n_alg):
            b = algos[j]
            x = piv[a].to_numpy(dtype=float)
            y = piv[b].to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            x, y = x[mask], y[mask]
            diff = x - y
            try:
                p = 1.0 if len(x) < 5 or np.allclose(diff, 0) else float(
                    stats.wilcoxon(x, y, zero_method="wilcox").pvalue
                )
            except ValueError:
                p = 1.0
            raw[i, j] = raw[j, i] = p
            pairs.append((p, i, j))

    # Holm's step-down adjustment over all pairwise tests in this cell.
    adjusted = np.ones((n_alg, n_alg))
    running = 0.0
    m = len(pairs)
    for rank, (p, i, j) in enumerate(sorted(pairs), start=1):
        running = max(running, (m - rank + 1) * p)
        adjusted[i, j] = adjusted[j, i] = min(1.0, running)

    sym = np.full((n_alg, n_alg), r"$\cdot$", dtype=object)
    for i, a in enumerate(algos):
        sym[i, i] = "---"
        for j in range(i + 1, n_alg):
            b = algos[j]
            if adjusted[i, j] > 0.05:
                mark_ab = mark_ba = r"$\approx$"
            else:
                mark_ab = "+" if piv[a].mean() > piv[b].mean() else "-"
                mark_ba = "+" if piv[b].mean() > piv[a].mean() else "-"
            sym[i, j] = mark_ab
            sym[j, i] = mark_ba

    lines = [tab_begin("l " + " ".join(["c"] * len(algos))), r"\toprule",
             " & " + " & ".join(short(a) for a in algos) + r" \\", r"\midrule"]
    for i, a in enumerate(algos):
        lines.append(short(a) + " & " + " & ".join(sym[i]) + r" \\")
    lines += [r"\bottomrule", tab_end()]
    write_tex(outfile, "\n".join(lines))
    pd.DataFrame(adjusted, index=algos, columns=algos).to_csv(
        OUT_TAB / outfile.replace(".tex", "_p.csv")
    )
    # Numbered Holm-adjusted p-value companion table.
    p_tex = outfile.replace(".tex", "p.tex")
    plines = [tab_begin("l " + " ".join(["c"] * len(algos))), r"\toprule",
              " & " + " & ".join(short(a) for a in algos) + r" \\", r"\midrule"]
    for i, a in enumerate(algos):
        cells = []
        for j, _b in enumerate(algos):
            if i == j:
                cells.append("---")
            else:
                pv = adjusted[i, j]
                if not np.isfinite(pv):
                    cells.append("---")
                elif pv < 1e-3:
                    cells.append(f"{pv:.1e}")
                else:
                    cells.append(f"{pv:.3f}")
        plines.append(short(a) + " & " + " & ".join(cells) + r" \\")
    plines += [r"\bottomrule", tab_end()]
    write_tex(p_tex, "\n".join(plines))
    print(
        label,
        "Holm-adjusted MA vs Random p=",
        adjusted[algos.index("MA"), algos.index("Random")]
        if "MA" in algos and "Random" in algos else "n/a",
    )


def build_wilcoxon():
    e2 = pd.read_csv(RAW / "e2_sparsity_full.csv")
    e3 = load_e3_submission()
    pairwise_matrix(e2[e2.sparsity == 0.5], "E2 f=0.5", "table_s7.tex")
    pairwise_matrix(e3[e3.dim == 16], "E3 dim16", "table_s8.tex")


# ---------- Part E N* ----------
def fig_s2_unstable_nstar():
    """Observed FHR curves under target splits, with unstable N* summarized below."""
    plt.rcParams.update(SUP_RC)
    e1 = load_e1_submission()
    nh = e1[e1["dataset"].astype(str).str.contains("nhanes", case=False, na=False)]
    scales = np.array(sorted(nh["scale"].unique()), dtype=float)
    x = np.log10(scales)
    xlim = (float(x.min()) - 0.12, float(x.max()) + 0.12)

    def logistic(xv, L, k, x0):
        return L / (1.0 + np.exp(-k * (xv - x0)))

    targets = sorted(nh["target_id"].unique())
    rng = np.random.default_rng(0)
    algos_show = ["MA", "NSGA2", "SPEA2", "Random"]
    colors = {"MA": "#1b9e77", "NSGA2": "#d95f02", "SPEA2": "#7570b3", "Random": "#e7298a"}
    markers = {"MA": "o", "NSGA2": "s", "SPEA2": "D", "Random": "^"}

    fig = plt.figure(figsize=(7.2, 6.2))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 0.95], hspace=0.42, wspace=0.22)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    nstar_records = []  # (split, algo, log10_nstar or nan, stable)

    for ax_i, ax in enumerate(axes):
        chosen = rng.choice(targets, size=min(8, len(targets)), replace=False)
        sub = nh[nh["target_id"].isin(chosen)]
        ss = seed_fhr_stats(sub, ["scale"])
        for algo in algos_show:
            ys, yerr = [], []
            for sc in scales:
                row = ss[(ss["scale"] == sc) & (ss["algorithm"] == algo)]
                if row.empty:
                    ys.append(np.nan)
                    yerr.append(np.nan)
                else:
                    ys.append(float(row["mean"].iloc[0]))
                    yerr.append(float(row["std"].iloc[0]))
            ys = np.asarray(ys, float)
            yerr = np.asarray(yerr, float)
            ax.errorbar(
                x, ys, yerr=yerr, fmt=markers[algo] + "-", ms=4.5, lw=1.15,
                color=colors[algo], label=short(algo), alpha=0.95, capsize=2,
            )
            mask = np.isfinite(ys)
            stable = False
            log_n = np.nan
            if mask.sum() >= 4 and (np.nanmax(ys) - np.nanmin(ys)) > 0.05:
                try:
                    from scipy.optimize import curve_fit
                    # Increasing/decreasing logistic; k can be +/-.
                    popt, _ = curve_fit(
                        logistic, x[mask], ys[mask],
                        p0=[float(np.nanmax(ys)), 1.0, float(np.nanmean(x))],
                        maxfev=8000, bounds=([0, -20, -10], [1.5, 20, 20]),
                    )
                    xx = np.linspace(xlim[0], xlim[1], 120)
                    ax.plot(xx, logistic(xx, *popt), ls="--", color=colors[algo], alpha=0.55, lw=1.0)
                    log_n = float(popt[2])
                    stable = xlim[0] <= log_n <= xlim[1]
                    if stable:
                        ax.axvline(log_n, color=colors[algo], ls=":", alpha=0.45, lw=0.9)
                except Exception:
                    pass
            nstar_records.append((ax_i + 1, short(algo), log_n, stable))
        ax.set_xlim(*xlim)
        ax.set_ylim(-0.05, 1.08)
        ax.set_title(f"Split {ax_i + 1} ({len(chosen)}/{len(targets)} targets)")
        ax.grid(True, ls=":", alpha=0.35)
        if ax_i % 2 == 0:
            ax.set_ylabel("Mean FHR")
        if ax_i >= 2:
            ax.set_xlabel(r"$\log_{10} N$")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=4, frameon=False,
        bbox_to_anchor=(0.5, 0.995), columnspacing=1.2, handlelength=2.2,
    )

    # Bottom panel: fitted log10(N*) across splits; shade observed-data band.
    axb = fig.add_subplot(gs[2, :])
    xpos0 = np.arange(len(algos_show))
    y_lo, y_hi = xlim[0] - 0.55, xlim[1] + 0.55
    for s_i in range(1, 5):
        for a_i, algo in enumerate(algos_show):
            recs = [r for r in nstar_records if r[0] == s_i and r[1] == short(algo)]
            if not recs:
                continue
            _, _, log_n, stable = recs[0]
            xloc = xpos0[a_i] + (s_i - 2.5) * 0.16
            if not np.isfinite(log_n):
                axb.scatter(xloc, y_lo + 0.08, marker="x", s=32, color=colors[algo], alpha=0.85, zorder=3)
                continue
            # Clip extreme unstable estimates to the axis edge so every split is visible.
            y_plot = float(np.clip(log_n, y_lo + 0.05, y_hi - 0.05))
            axb.scatter(
                xloc, y_plot,
                s=38, color=colors[algo],
                marker="o" if stable else "x",
                alpha=0.9, zorder=3,
            )
    axb.axhspan(xlim[0], xlim[1], color="#bbbbbb", alpha=0.25)
    axb.set_xticks(xpos0)
    axb.set_xticklabels([short(a) for a in algos_show])
    axb.set_ylabel(r"fitted $\log_{10} N^*$")
    axb.set_xlabel(r"Algorithm (4 markers/algo = splits; $\circ$=in-range, $\times$=unstable/failed, clipped to axis)")
    axb.set_ylim(y_lo, y_hi)
    axb.set_xlim(-0.6, len(algos_show) - 0.4)
    axb.grid(True, axis="y", ls=":", alpha=0.35)
    axb.set_title(r"Fitted $\log_{10} N^*$ vs.\ observed scale range (gray band)")

    fig.suptitle(
        r"Exploratory logistic fits are split-dependent and often unstable",
        y=1.01, fontsize=10,
    )
    for d in (OUT_FIG, FIGS):
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / "fig_s2_unstable_nstar.pdf", bbox_inches="tight", pad_inches=0.05)
        fig.savefig(d / "fig_s2_unstable_nstar.png", dpi=220, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print("fig S2 done")


def fig_s3_ffg():
    """FFG distributions; hit counts go in the LaTeX caption (not drawn under axes)."""
    plt.rcParams.update(SUP_RC)
    e2 = pd.read_csv(RAW / "e2_sparsity_full.csv")
    hard = e2[e2.sparsity == 0.5].copy()
    algos = ["MA", "NSGA2", "RandomRepair"]
    fig, ax = plt.subplots(figsize=(5.8, 3.3))
    data, labels, notes = [], [], []
    for a in algos:
        g = hard[hard.algorithm == a]
        hit = g[g["FHR"] >= 0.5]["FFG_evals"].to_numpy()
        n_succ = int((g["FHR"] >= 0.5).sum())
        n_tot = len(g)
        data.append(hit if len(hit) else np.array([np.nan]))
        labels.append(short(a))
        med = float(np.median(hit)) if len(hit) else np.nan
        notes.append(
            f"{short(a)}: {n_succ}/{n_tot}, med={med:.0f}" if len(hit) else f"{short(a)}: 0/{n_tot}"
        )
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor("#c6dbef")
    rng = np.random.default_rng(0)
    for i, d in enumerate(data, start=1):
        d = d[np.isfinite(d)]
        if len(d) == 0:
            continue
        jitter = rng.normal(0, 0.06, size=len(d))
        ax.scatter(np.full(len(d), i) + jitter, d, s=9, alpha=0.35, color="#08519c", zorder=3)
    ax.set_ylabel("FFG evaluations (hits only)")
    ax.set_xlabel("Algorithm")
    ax.set_title(r"E2 $f=0.5$: evaluations to first feasible")
    ax.set_yscale("log")
    ax.grid(True, axis="y", ls=":", alpha=0.4)
    fig.tight_layout(pad=0.35)
    # Persist hit-count note for the TeX caption helper.
    note_path = OUT_TAB / "fig_s3_hit_notes.txt"
    OUT_TAB.mkdir(parents=True, exist_ok=True)
    note_path.write_text("; ".join(notes))
    for d in (OUT_FIG, FIGS):
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / "fig_s3_ffg_first_hit.pdf", bbox_inches="tight", pad_inches=0.03)
        fig.savefig(d / "fig_s3_ffg_first_hit.png", dpi=220, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("fig S3 done", notes)


def build_s9():
    rows_data = []
    campaigns = {
        "E1": load_e1_submission(),
        "E2": pd.read_csv(RAW / "e2_sparsity_full.csv"),
        "E3": load_e3_submission(),
    }
    for camp, df in campaigns.items():
        df = df[df["algorithm"].isin(ALG_ORDER)]
        med = df.groupby("algorithm")["n_evals"].median()
        for a, v in med.items():
            rows_data.append({"campaign": camp, "algorithm": a, "median_n_evals": v})
    wide = pd.DataFrame(rows_data).pivot(index="algorithm", columns="campaign", values="median_n_evals")
    for c in ["E1", "E2", "E3"]:
        if c not in wide.columns:
            wide[c] = np.nan
    wide = wide[["E1", "E2", "E3"]]
    algos = [a for a in ALG_ORDER if a in wide.index]
    lines = [tabx_begin(r">{\raggedright\arraybackslash}X ccc"), r"\toprule",
             r"Algorithm & E1 median $n_{\mathrm{evals}}$ & E2 & E3 \\", r"\midrule",
             r"Theoretical ceiling & 100000 & 100000 & 100000 \\", r"\midrule"]
    for a in algos:
        e1 = wide.loc[a, "E1"]; e2 = wide.loc[a, "E2"]; e3 = wide.loc[a, "E3"]
        def f(x):
            return "---" if not np.isfinite(x) else f"{x:.0f}"
        lines.append(f"{short(a)} & {f(e1)} & {f(e2)} & {f(e3)} \\\\")
    lines += [r"\bottomrule", tabx_end()]
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
